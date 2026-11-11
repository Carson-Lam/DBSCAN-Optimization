"""
run_world_poi.py
================
Real-data experiment: grid top-k DBSCAN vs. full indexed DBSCAN on the
World-POI dataset https://github.com/onspatial/world-poi.

What it does:
-----
  1. Load fsq_latitude / fsq_longitude (and fsq_place_id for de-duplication)
     from the 7.8 GB CSV in chunks. 

  2. Project lat/lon to km (see "Projection" below).

  3. Build one Indexicon R-tree and one hierarchical grid, shared by every run.

  4. For each eps: grid top-k for every k (and refine setting). 

  5. For each eps: the full DBSCAN_indexed baseline. Slow, possibly 24 h+.
     Progress and an ETA go to the log every 1%. If it is stopped (Ctrl-C or
     --baseline-hours), it is recorded as "did not finish".

  6. Check the top-k results against the baseline: every returned cluster must
     have exactly the same core points as one baseline cluster (exactness), and
     we count how many are among the true k densest (fidelity).

Usage
-----
  # 1) quick look at neighbourhood sizes to pick eps / minPts :
  python run_world_poi.py World_POI_trigrams_0.5.csv --probe

  # 2) small run on one city (New York as an example):
  python run_world_poi.py World_POI_trigrams_0.5.csv --bbox=-74.3,40.5,-73.7,40.95

  # 3) long run on all data points
  python run_world_poi.py World_POI_trigrams_0.5.csv --eps 0.1 --minpts 20

Everything is written to results/world_poi/<tag>/ :
  run.log        
  summary.json    settings, dataset stats, index costs, baseline status
  topk.csv        time, range queries, exactness, fidelity
  baseline_eps<eps>.npy   

Projection
----------
- DBSCAN needs a 2D plane.

- We slice the globe into 60 vertical strips 6° longitude wide. We then

- flatten each strip along its own center meridian (lon_center) using
    x = R * (lon - lon_center) * cos(lat),  y = R * lat.
  The cos(lat) factor is to make the degrees shorter towards the poles like a globe.

- Strips are laid side-by-side with gaps of at least 333km to avoid a point
  in the strips being within eps of a point in another

- Longitude wraps around: the 180th meridian (Russia / Alaska, Fiji) 
  is not an edge of the map.

*** A cluster on a strip edge can get split in two, but the script tries to 
    minimize this by picking edges near the fewest points.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from algorithms import DBSCAN_grid, DBSCAN_indexed
from grid_index import HierarchicalGridIndex
from spatial_index import IndexiconIndex

R_EARTH_KM = 6371.0088          # mean Earth radius
ZONE_SPACING_KM = 1000.0        # zone centres along x; a zone is <= 667 km wide
LAT_COL, LON_COL, ID_COL = "fsq_latitude", "fsq_longitude", "fsq_place_id"


# ---------------------------------------------------------------------------
# logging and memory
# ---------------------------------------------------------------------------
class Log:
    """print() to the screen and to run.log at the same time."""

    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, msg=""):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def rss_gb():
    """Current memory use of this process, in GB (0 if unknown)."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**30
    except ImportError:
        pass
    try:
        import resource                      # Linux / macOS only
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return kb / 2**20 if sys.platform != "darwin" else kb / 2**30
    except ImportError:
        return 0.0


def hms(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


# ---------------------------------------------------------------------------
# 1. loading
# ---------------------------------------------------------------------------
def load_csv(path, dedupe, log, chunk_rows=2_000_000):
    """Read lat, lon (and place ids) from the CSV. Returns lat, lon arrays and
    a dict of counts for the summary."""
    import pandas as pd

    # Tab seperation bugfix
    with open(path, encoding="utf-8", errors="replace") as f:
        first = f.readline()
    if first.count("\t") > first.count(","):
        fmt = dict(sep="\t", quoting=csv.QUOTE_NONE)
        log("  tab-separated file detected")
    else:
        fmt = dict(sep=",")
        log("  comma-separated file detected")

    header = pd.read_csv(path, nrows=0, **fmt).columns.tolist()
    missing = [c for c in (LAT_COL, LON_COL) if c not in header]
    if missing:
        sys.exit(f"Columns {missing} not in the CSV. It has: {header}")
    if dedupe and ID_COL not in header:
        sys.exit(f"No {ID_COL} column to de-duplicate on. Run with --no-dedupe.")
    cols = [LAT_COL, LON_COL] + ([ID_COL] if dedupe else [])

    lats, lons, hashes = [], [], []
    rows = 0
    t0 = time.time()
    reader = pd.read_csv(path, usecols=cols, chunksize=chunk_rows, dtype=str,
                         on_bad_lines="skip", low_memory=False, **fmt)
    for chunk in reader:
        rows += len(chunk)
        lats.append(pd.to_numeric(chunk[LAT_COL], errors="coerce").to_numpy(np.float64))
        lons.append(pd.to_numeric(chunk[LON_COL], errors="coerce").to_numpy(np.float64))
        if dedupe:
            # 64-bit hashes instead of the id strings for far less memory
            ids = chunk[ID_COL].astype("string").fillna("")
            hashes.append(pd.util.hash_pandas_object(ids, index=False).to_numpy())
        log(f"  read {rows:,} rows  ({time.time() - t0:.0f}s, {rss_gb():.1f} GB)")

    log(f"  {rows:,} rows read (the README lists 8,744,036 for "
        f"World_POI_trigrams_0.5; a big gap means lines were skipped as malformed)")
    lat = np.concatenate(lats)
    lon = np.concatenate(lons)
    stats = {"csv_rows": rows}

    ok = (np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) <= 90)
          & (np.abs(lon) <= 180) & ~((lat == 0) & (lon == 0)))   # (0,0) = missing
    stats["dropped_invalid"] = int((~ok).sum())

    if dedupe:
        h = np.concatenate(hashes)
        _, first = np.unique(h, return_index=True)               # first row per place
        keep = np.zeros(len(h), dtype=bool)
        keep[first] = True
        stats["dropped_duplicate_place_id"] = int((ok & ~keep).sum())
        ok &= keep

    lat, lon = lat[ok], lon[ok]
    stats["points"] = int(len(lat))
    log(f"  {stats}")
    return lat, lon, stats


# ---------------------------------------------------------------------------
# 2. projection
# ---------------------------------------------------------------------------
def best_zone_offset(lon, lat, width, margin_km):
    """Shift (in degrees) of the zone edges that puts the fewest points within
    margin_km of an edge. Edges sit at offset + width * j."""
    # a point is near an edge if its east-west distance to it is < margin_km;
    # in degrees of longitude that is margin / (R cos lat)
    cos = np.maximum(np.cos(np.radians(lat)), 0.05)
    tol_deg = np.degrees(margin_km / (R_EARTH_KM * cos))
    best, best_n = 0.0, None
    for off in np.arange(0.0, width, 0.1):
        d = np.abs(((lon - off) + width / 2) % width - width / 2)   # deg to nearest edge
        n = int((d < tol_deg).sum())
        if best_n is None or n < best_n:
            best, best_n = round(float(off), 1), n
    return best, best_n


def project_zones(lat, lon, width, offset):
    """Zone-wise sinusoidal projection, in km

    Longitude is treated as a circle: zones are numbered modulo 360 degrees,
    so the zone that contains the 180th meridian is one zone, and points just
    east and just west of it (e.g. eastern Russia and Alaska) get neighbouring
    x coordinates, not opposite ends of the map.

    """
    n_zones = round(360 / width)                             # 60 strips around the globe
    zone = np.floor((lon - offset) / width) % n_zones        # which strip is the point in? (positive)
    lon0 = offset + (zone + 0.5) * width                     # strip's center line
    dlon = (lon - lon0 + 180.0) % 360.0 - 180.0              # How far E/W from center line is the pt? 
                                                             #     Using +180, % 360, -180 goes the short way
    phi = np.radians(lat)                                                       
    x = R_EARTH_KM * np.radians(dlon) * np.cos(phi) + zone * ZONE_SPACING_KM
    y = R_EARTH_KM * phi
    return np.column_stack([x, y]), zone


def project_local(lat, lon):
    """One region: equirectangular around the region's centre latitude, in km.
    Fine for a city or a small country."""
    lat0 = np.radians(np.median(lat))
    x = R_EARTH_KM * np.radians(lon - np.median(lon)) * np.cos(lat0)
    y = R_EARTH_KM * np.radians(lat - np.median(lat))
    return np.column_stack([x, y])


def load_points(args, log):
    """Steps 1 + 2, with a cache so the CSV is only parsed once."""
    key = json.dumps({"csv": str(Path(args.csv).resolve()),
                      "size": os.path.getsize(args.csv),
                      "dedupe": not args.no_dedupe}, sort_keys=True)
    cache = Path(args.cache_dir) / f"latlon_{hashlib.md5(key.encode()).hexdigest()[:10]}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        lat, lon = z["lat"], z["lon"]
        stats = json.loads(str(z["stats"]))
        log(f"loaded {len(lat):,} points from cache {cache}")
    else:
        log(f"reading {args.csv} (first run only; cached afterwards)")
        lat, lon, stats = load_csv(args.csv, not args.no_dedupe, log)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, lat=lat, lon=lon, stats=json.dumps(stats))
        log(f"cached to {cache}")

    if args.bbox:
        x0, y0, x1, y1 = args.bbox
        m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        lat, lon = lat[m], lon[m]
        log(f"bbox {args.bbox}: {len(lat):,} points")
    if args.sample and args.sample < len(lat):
        pick = np.sort(np.random.default_rng(0).choice(len(lat), args.sample, replace=False))
        lat, lon = lat[pick], lon[pick]
        log(f"random sample of {len(lat):,} points (seed 0)")
    if len(lat) == 0:
        sys.exit("No points left after filtering.")

    proj = {}
    if args.bbox:
        xy = project_local(lat, lon)
        proj["method"] = "local equirectangular"
    else:
        margin = max(args.eps)
        off, near = best_zone_offset(lon, lat, args.zone_width, margin)
        xy, zone = project_zones(lat, lon, args.zone_width, off)
        proj.update(method="zoned sinusoidal", zone_width_deg=args.zone_width,
                    zone_offset_deg=off, zones_used=int(len(np.unique(zone))),
                    points_within_eps_of_zone_edge=near)
        log(f"projection: {args.zone_width} deg zones, edges at {off} + "
            f"{args.zone_width}j deg; {near:,} points ({100 * near / len(xy):.3f}%) "
            f"are within {margin} km of a zone edge")
    stats.update(points_used=int(len(xy)), projection=proj)
    return np.ascontiguousarray(xy), stats


# ---------------------------------------------------------------------------
# probe: neighbourhood sizes, to choose eps / minPts
# ---------------------------------------------------------------------------
def probe(xy, eps_list, log, n_sample=20_000):
    from scipy.spatial import cKDTree
    t0 = time.time()
    kd = cKDTree(xy)
    rng = np.random.default_rng(0)
    q = xy[rng.choice(len(xy), min(n_sample, len(xy)), replace=False)]
    log(f"probe: neighbourhood sizes of {len(q):,} random points "
        f"(point itself included), cKDTree built in {time.time() - t0:.0f}s")
    log(f"  {'eps km':>7} {'p50':>7} {'p90':>7} {'p99':>7} {'max':>8}   "
        f"share of points that are core for minPts = 5 / 10 / 20 / 50 / 100")
    for eps in eps_list:
        c = kd.query_ball_point(q, eps, return_length=True, workers=-1)
        p50, p90, p99 = np.percentile(c, [50, 90, 99])
        core = " / ".join(f"{100 * np.mean(c >= m):4.1f}%" for m in (5, 10, 20, 50, 100))
        log(f"  {eps:>7g} {p50:>7.0f} {p90:>7.0f} {p99:>7.0f} {c.max():>8,}   {core}")
    log("Pick minPts so the core share is small (a few %): clusters are then the "
        "dense hot spots rather than whole countries.")


# ---------------------------------------------------------------------------
# 6. checking top-k against the baseline
# ---------------------------------------------------------------------------
def check_topk(grid_labels, base_labels, counts, minPts, k):
    """Compare a top-k labelling with the full baseline.

    counts:   eps-neighbourhood size of every point 

    exact:    every returned cluster has exactly the core points of one baseline
              cluster

    hits:     returned clusters that are among the k baseline clusters with the
              highest peak density (largest neighbourhood of any core point).
              Ties at the k-th place count as hits.
    """
    core = counts >= minPts
    g, b = grid_labels[core], base_labels[core]
    sel = g != -1
    pairs = np.unique(np.stack([g[sel], b[sel]], axis=1), axis=0)
    out = {"returned": int(len(np.unique(g[sel])))}

    one_to_one = (len(np.unique(pairs[:, 0])) == len(pairs) ==
                  len(np.unique(pairs[:, 1])) and (pairs[:, 1] != -1).all())
    sizes_ok = True
    if one_to_one and len(pairs):
        bsize = np.bincount(b[b != -1])
        gsize = np.bincount(g[sel])
        sizes_ok = all(gsize[gl] == bsize[bl] for gl, bl in pairs)
    out["exact"] = bool(one_to_one and sizes_ok)

    # peak density per baseline cluster
    peak = np.zeros(int(base_labels.max()) + 1)
    np.maximum.at(peak, b[b != -1], counts[core][b != -1])
    if len(pairs) and k <= (peak > 0).sum():
        kth = np.sort(peak)[::-1][k - 1]
        got = pairs[:, 1]
        out["hits"] = int((peak[got] >= kth).sum())
        order = np.argsort(-peak, kind="stable")
        rank = np.empty_like(order)
        rank[order] = np.arange(1, len(order) + 1)
        out["ranks"] = sorted(int(rank[x]) for x in got)
        # k=1 also checks order: was the first cluster found the densest one?
        first = pairs[pairs[:, 0] == pairs[:, 0].min(), 1][0]
        out["first_is_densest"] = bool(peak[first] >= peak.max())
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv", help="path to World_POI_trigrams_0.5.csv")
    p.add_argument("--eps", type=float, nargs="+", default=[0.1],
                   help="eps values in km (default 0.1). One baseline run each.")
    p.add_argument("--minpts", type=int, default=20)
    p.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10, 25, 100])
    p.add_argument("--refine", type=int, nargs="+", default=[1, 8],
                   help="refine settings for DBSCAN_grid (default 1 and 8)")
    p.add_argument("--probe", action="store_true",
                   help="only print neighbourhood-size statistics, then exit")
    p.add_argument("--bbox", type=lambda s: [float(v) for v in s.split(",")],
                   help="minlon,minlat,maxlon,maxlat: one region instead of the world. Use --bbox=... when it starts with a minus sign")
    p.add_argument("--sample", type=int, help="random subset of this many points")
    p.add_argument("--no-dedupe", action="store_true",
                   help="keep every CSV row (one place can appear in several rows)")
    p.add_argument("--zone-width", type=float, default=6.0,
                   help="longitude zone width in degrees; must divide 360 (default 6)")
    p.add_argument("--no-baseline", action="store_true", help="grid top-k only")
    p.add_argument("--baseline-hours", type=float,
                   help="stop the baseline after this long and record DNF "
                        "(default: no limit)")
    p.add_argument("--grid-full", action="store_true",
                   help="also run DBSCAN_grid with no k limit (as slow as the baseline)")
    p.add_argument("--tag", help="name of the results folder (default: timestamp)")
    p.add_argument("--out", default="results/world_poi")
    p.add_argument("--cache-dir", default="cache")
    return p.parse_args()


class _OutOfTime(Exception):
    pass


def main():
    args = parse_args()
    if abs(360 / args.zone_width - round(360 / args.zone_width)) > 1e-9:
        sys.exit("--zone-width must divide 360 evenly (e.g. 3, 4, 6, 10)")
    tag = args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out) / tag
    out.mkdir(parents=True, exist_ok=True)
    log = Log(out / "run.log")
    log(f"run {tag}: {' '.join(sys.argv)}")

    summary = {"tag": tag, "argv": sys.argv, "settings": vars(args)}
    def save_summary():
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # 1-2. points
    t0 = time.time()
    xy, summary["data"] = load_points(args, log)
    n = len(xy)
    log(f"{n:,} points ready in {time.time() - t0:.0f}s, extent "
        f"{np.ptp(xy[:, 0]):,.0f} x {np.ptp(xy[:, 1]):,.0f} km, {rss_gb():.1f} GB")
    save_summary()
    if args.probe:
        probe(xy, sorted(set(args.eps) | {0.05, 0.1, 0.2, 0.5}), log)
        return

    # 3. indexes
    t0 = time.time()
    tree = IndexiconIndex(xy)                       # R-tree
    t_tree = time.time() - t0
    log(f"Indexicon R-tree: {t_tree:.1f}s, {tree.nbytes() / 2**30:.2f} GB")
    t0 = time.time()
    grid = HierarchicalGridIndex(xy, eps_min=min(args.eps), eps_max=max(args.eps))
    t_grid = time.time() - t0
    log(f"hierarchical grid: {t_grid:.1f}s, {grid.nbytes() / 2**30:.2f} GB, "
        f"{len(grid.sides)} rungs\n{grid.describe()}")
    summary["index"] = {"rtree_s": t_tree, "rtree_gb": tree.nbytes() / 2**30,
                        "grid_s": t_grid, "grid_gb": grid.nbytes() / 2**30,
                        "grid_rungs": len(grid.sides), "rss_gb": rss_gb()}
    save_summary()

    topk_rows = []
    topk_fields = ["eps_km", "minPts", "k", "refine", "seconds", "range_queries",
                   "clusters_returned", "largest_sizes", "exact", "hits",
                   "hit_rate", "ranks", "first_is_densest"]
    def save_topk():
        with open(out / "topk.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=topk_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(topk_rows)

    # 4. grid top-k (fast)
    topk_labels = {}
    for eps in args.eps:
        for refine in args.refine:
            for k in sorted(args.k):
                t0 = time.time()
                lab, _, _, rq = DBSCAN_grid(xy, eps, args.minpts, max_iterations=k,
                                            tree=tree, grid=grid, refine=refine)
                dt = time.time() - t0
                sizes = np.bincount(lab[lab > 0])
                sizes = sorted(sizes[sizes > 0].tolist(), reverse=True)
                row = {"eps_km": eps, "minPts": args.minpts, "k": k, "refine": refine,
                       "seconds": round(dt, 3), "range_queries": rq,
                       "clusters_returned": len(sizes),
                       "largest_sizes": " ".join(map(str, sizes[:5]))}
                topk_rows.append(row)
                topk_labels[(eps, refine, k)] = lab.astype(np.int32)
                log(f"grid eps={eps} refine={refine} top-{k}: {dt:.2f}s, "
                    f"{rq:,} range queries, sizes {sizes[:5]}")
                save_topk()

    if args.grid_full:
        for eps in args.eps:
            t0 = time.time()
            lab, _, _, rq = DBSCAN_grid(xy, eps, args.minpts, tree=tree, grid=grid)
            log(f"grid eps={eps} FULL run: {hms(time.time() - t0)}, {rq:,} range "
                f"queries, {int(lab.max())} clusters")
            summary.setdefault("grid_full", {})[str(eps)] = {
                "seconds": time.time() - t0, "range_queries": rq,
                "clusters": int(lab.max())}
            np.save(out / f"gridfull_eps{eps}.npy", lab.astype(np.int32))
            save_summary()

    if args.no_baseline:
        log("--no-baseline: done.")
        return

    # 5. baseline (slow)
    from scipy.spatial import cKDTree
    kd = None
    summary["baseline"] = {}
    for eps in args.eps:
        state = {"t0": time.time(), "last": 0.0}

        def progress(done, total, clusters, rq):
            el = time.time() - state["t0"]
            state.update(done=done, clusters=clusters, rq=rq, el=el)
            if done and el - state["last"] >= 60:        # at most one line a minute
                state["last"] = el
                eta = el / max(rq, 1) * (total - rq)      # ~one range query per point
                log(f"  baseline eps={eps}: {100 * done / total:5.1f}% of points, "
                    f"{rq:,} range queries, {clusters:,} clusters, {hms(el)} elapsed, "
                    f"ETA ~{hms(eta)}, {rss_gb():.1f} GB")
            if args.baseline_hours and el > args.baseline_hours * 3600:
                raise _OutOfTime

        log(f"baseline eps={eps}: full DBSCAN_indexed on {n:,} points "
            f"(progress every minute)")
        status = {}
        interrupted = False
        try:
            base, _, rq = DBSCAN_indexed(xy, eps, args.minpts, tree=tree,
                                         progress=progress)
            dt = time.time() - state["t0"]
            status = {"finished": True, "seconds": dt, "range_queries": rq,
                      "clusters": int(base.max()),
                      "noise": int((base == -1).sum())}
            log(f"baseline eps={eps} finished: {hms(dt)}, {rq:,} range queries, "
                f"{int(base.max()):,} clusters")
            np.save(out / f"baseline_eps{eps}.npy", base.astype(np.int32))
        except (_OutOfTime, KeyboardInterrupt) as e:
            status = {"finished": False, "reason": type(e).__name__,
                      "seconds": state.get("el"), "points_visited": state.get("done"),
                      "range_queries": state.get("rq"), "clusters": state.get("clusters")}
            log(f"baseline eps={eps} DID NOT FINISH ({type(e).__name__}) after "
                f"{hms(state.get('el', 0))}: {status}")
            base = None
            interrupted = isinstance(e, KeyboardInterrupt)
        summary["baseline"][str(eps)] = status
        save_summary()
        if base is None:
            if interrupted:
                break
            continue

        # 6. exactness and fidelity of every top-k run at this eps
        t0 = time.time()
        if kd is None:
            kd = cKDTree(xy)
        counts = kd.query_ball_point(xy, eps, return_length=True, workers=-1)
        log(f"neighbour counts from cKDTree: {time.time() - t0:.0f}s")
        for row in topk_rows:
            if row["eps_km"] != eps:
                continue
            k = row["k"]
            chk = check_topk(topk_labels[(eps, row["refine"], k)], base, counts,
                             args.minpts, k)
            row.update(exact=chk["exact"], hits=chk.get("hits"),
                       hit_rate=None if chk.get("hits") is None else chk["hits"] / k,
                       ranks=" ".join(map(str, chk.get("ranks", [])[:25])),
                       first_is_densest=chk.get("first_is_densest"))
            row["speedup_vs_baseline"] = round(status["seconds"] / max(row["seconds"], 1e-9), 1)
            log(f"  check eps={eps} refine={row['refine']} top-{k}: exact={chk['exact']} "
                f"hits={chk.get('hits')}/{k} speedup {row['speedup_vs_baseline']:,.0f}x")
        if "speedup_vs_baseline" not in topk_fields:
            topk_fields.append("speedup_vs_baseline")
        save_topk()

    log("done.")


if __name__ == "__main__":
    main()