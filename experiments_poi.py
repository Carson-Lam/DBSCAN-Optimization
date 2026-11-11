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

  # 4) paper figures 5, 6, 7 and the index table (the default --exp).
  python experiments_poi.py World_POI_trigrams_0.5.csv --eps 0.1 --minpts 20 --tag paper
  #    resume after a crash: the same command (same --tag) skips finished rows
  #    redraw figures only:  python experiments_poi.py X.csv --tag paper --plot-only

  Figure outputs (same folder): fig5_scaling.{csv,png,pdf},
  fig6_topk_quality.{csv,png,pdf}, fig7_topk_map.{png,pdf} + fig7_clusters.csv,
  table_indexes.{csv,md,tex}

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

from algorithms import DBSCAN_grid, DBSCAN_indexed, DBSCAN_single_maxrs
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

    if not args.keep_file_order:
        # The file is grouped by country, so vanilla DBSCAN (which visits
        # points in input order) would find its first clusters wherever the
        # file starts. A fixed random order removes that artefact, and the
        # first n points of it are the n-point sample used in every experiment
        # (so the samples are nested: 1K inside 10K inside 100K ...).
        perm = np.random.default_rng(0).permutation(len(lat))
        lat, lon = lat[perm], lon[perm]
        log("points shuffled (seed 0)")

    if args.bbox:
        x0, y0, x1, y1 = args.bbox
        m = (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)
        lat, lon = lat[m], lon[m]
        log(f"bbox {args.bbox}: {len(lat):,} points")
    if args.sample and args.sample < len(lat):
        lat, lon = lat[:args.sample], lon[:args.sample]
        log(f"first {len(lat):,} points of the shuffled order")
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
    return np.ascontiguousarray(xy), stats, lat, lon


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
    p.add_argument("--exp", nargs="*", default=["fig5", "fig6", "fig7", "table"],
                   choices=["fig5", "fig6", "fig7", "table", "check"],
                   help="experiments to run, in this order (default: fig5 fig6 "
                        "fig7 table). 'check' is the top-k exactness/fidelity run.")
    p.add_argument("--sizes", type=int, nargs="+",
                   default=[1_000, 10_000, 100_000, 1_000_000, 0],
                   help="point counts for fig5 and the table; 0 = all points")
    p.add_argument("--fig-k", type=int, default=3,
                   help="k for fig5, fig7 and the table (default 3)")
    p.add_argument("--fig6-kmax", type=int, default=20,
                   help="fig6 x-axis runs k = 1..this (default 20)")
    p.add_argument("--fig6-n", type=int, default=0, help="points for fig6 (0 = all)")
    p.add_argument("--fig7-n", type=int, default=0, help="points for fig7 (0 = all)")
    p.add_argument("--fig-refine", type=int, default=1,
                   help="refine setting of Grid-DBSCAN in the figures (default 1)")
    p.add_argument("--no-maxrs", action="store_true",
                   help="leave single-run MaxRS out of fig5 and fig6")
    p.add_argument("--indexes", nargs="+", default=list(IndexiconIndex.KINDS),
                   choices=list(IndexiconIndex.KINDS),
                   help="Indexicon structures for the table (default: all 7)")
    p.add_argument("--repeats", type=int, default=3,
                   help="timing repeats for runs under 1M points; the median is "
                        "reported (default 3)")
    p.add_argument("--keep-file-order", action="store_true",
                   help="do not shuffle the points (vanilla then visits them "
                        "in file order, which is grouped by country)")
    p.add_argument("--plot-only", action="store_true",
                   help="only redraw the figures from the CSVs in --tag's folder")
    p.add_argument("--tag", help="name of the results folder (default: timestamp). "
                        "Re-using a tag resumes: finished measurements are skipped")
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
    if args.plot_only:
        make_plots(out, log)
        return

    summary = {"tag": tag, "argv": sys.argv, "settings": vars(args)}
    def save_summary():
        (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # 1-2. points
    t0 = time.time()
    xy, summary["data"], lat, lon = load_points(args, log)
    n = len(xy)
    log(f"{n:,} points ready in {time.time() - t0:.0f}s, extent "
        f"{np.ptp(xy[:, 0]):,.0f} x {np.ptp(xy[:, 1]):,.0f} km, {rss_gb():.1f} GB")
    save_summary()
    if args.probe:
        probe(xy, sorted(set(args.eps) | {0.05, 0.1, 0.2, 0.5}), log)
        return

    for exp in args.exp:
        log(f"===== experiment: {exp} =====")
        if exp == "check":
            check_experiment(args, xy, log, out, summary, save_summary)
        else:
            EXPERIMENTS[exp](args, xy, lat, lon, log, out)
    if "fig5" in args.exp:
        log("===== fig5: full vanilla runs (slowest, last) =====")
        fig5_full(args, xy, log, out)
    if args.exp:
        make_plots(out, log)
    log("done.")


def check_experiment(args, xy, log, out, summary, save_summary):
    """Top-k exactness and fidelity against the full baseline (the original
    run of this script)."""
    n = len(xy)

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

    log("check: done.")

# ===========================================================================================
# ===========================================================================================
# ===========================================================================================

# Paper figures: fig5 (scaling), fig6 (top-k quality), fig7 (top-k map),
# table (Grid-DBSCAN with each Indexicon index)

# Names and styles in the plots. Change labels here to rename methods.
STYLE = {
    "vanilla_full":         dict(label="Vanilla DBSCAN + index (full run)",
                                 color="#2196F3", marker="o", ls="-"),
    "vanilla_full_largest": dict(label="Vanilla DBSCAN + index (full run, k largest)",
                                 color="#2196F3", marker="o", ls="-"),
    "vanilla_topk":         dict(label="Vanilla DBSCAN + index (stopped at k)",
                                 color="#9C27B0", marker="s", ls="--"),
    "maxrs_topk":           dict(label="MaxRS single-run (stopped at k)",
                                 color="#2E7D32", marker="D", ls=":"),
    "grid_topk":            dict(label="Grid-DBSCAN (stopped at k)",
                                 color="#FF5722", marker="^", ls="-"),
}


class Rows:
    """A CSV of results that is appended to row by row, so a crash loses at
    most the measurement in progress. Re-running with the same --tag skips
    rows that are already there."""

    def __init__(self, path, fields):
        self.path, self.fields = Path(path), fields
        self.rows = []
        if self.path.exists():
            with open(self.path, newline="") as f:
                self.rows = list(csv.DictReader(f))
        else:
            with open(self.path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def has(self, **key):
        return any(all(str(r.get(k)) == str(v) for k, v in key.items()) for r in self.rows)

    def add(self, **row):
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore").writerow(row)
        self.rows.append({k: str(v) for k, v in row.items()})


def _sizes(args, n_all):
    return sorted({n_all if s == 0 or s > n_all else s for s in args.sizes})


def _fmt_n(n):
    n = int(n)
    if n >= 1_000_000:
        return f"{n / 1e6:.1f}M".replace(".0M", "M")
    if n >= 1_000:
        return f"{n / 1e3:.0f}K"
    return str(n)


def _run(method, xy, eps, minPts, k, tree, grid, refine):
    """One run. Returns (labels, seconds, range_queries)."""
    t0 = time.perf_counter()
    if method == "grid_topk":
        lab, _, _, rq = DBSCAN_grid(xy, eps, minPts, max_iterations=k, tree=tree,
                                  grid=grid, refine=refine)
    elif method == "vanilla_topk":
        lab, _, rq = DBSCAN_indexed(xy, eps, minPts, max_iterations=k, tree=tree)
    elif method == "maxrs_topk":
        lab, _, _, rq = DBSCAN_single_maxrs(xy, eps, minPts, max_iterations=k, tree=tree)
    else:
        raise ValueError(method)
    return lab, time.perf_counter() - t0, rq


def _timed(method, xy, eps, minPts, k, tree, grid, refine, repeats):
    """Median time over `repeats` runs (1 run from 1M points up)."""
    reps = repeats if len(xy) < 1_000_000 else 1
    times = []
    for _ in range(max(reps, 1)):
        lab, dt, rq = _run(method, xy, eps, minPts, k, tree, grid, refine)
        times.append(dt)
    return lab, float(np.median(times)), rq


def _eps(args, log):
    if len(args.eps) > 1:
        log(f"figures use the first eps only ({args.eps[0]})")
    return args.eps[0]


def _build(xy, eps, log, kind="rtree"):
    t0 = time.perf_counter()
    tree = IndexiconIndex(xy, kind=kind)
    t_tree = time.perf_counter() - t0
    t0 = time.perf_counter()
    grid = HierarchicalGridIndex(xy, eps_min=eps, eps_max=eps)
    t_grid = time.perf_counter() - t0
    log(f"  n={len(xy):,}: {kind} {t_tree:.2f}s, grid {t_grid:.2f}s, {rss_gb():.1f} GB")
    return tree, grid, t_tree, t_grid


# ---------------------------------------------------------------------------
# Figure 5: runtime and range queries vs. number of points
# ---------------------------------------------------------------------------
FIG5_FIELDS = ["n", "method", "k", "seconds", "range_queries", "clusters",
               "finished", "note"]


def fig5_experiment(args, xy, lat, lon, log, out):
    """The fast methods at every size. The full vanilla runs come last
    (fig5_full), after every other experiment, because they can take a day."""
    eps, m, k = _eps(args, log), args.minpts, args.fig_k
    rows = Rows(out / "fig5_scaling.csv", FIG5_FIELDS)
    methods = ["grid_topk", "vanilla_topk"] + ([] if args.no_maxrs else ["maxrs_topk"])
    for n in _sizes(args, len(xy)):
        todo = [mt for mt in methods + ["build_rtree", "build_grid"]
                if not rows.has(n=n, method=mt)]
        if not todo:
            continue
        sub = xy[:n]
        tree, grid, t_tree, t_grid = _build(sub, eps, log)
        if "build_rtree" in todo:
            rows.add(n=n, method="build_rtree", seconds=t_tree, finished=True)
        if "build_grid" in todo:
            rows.add(n=n, method="build_grid", seconds=t_grid, finished=True)
        for mt in methods:
            if mt not in todo:
                continue
            lab, dt, rq = _timed(mt, sub, eps, m, k, tree, grid, args.fig_refine, args.repeats)
            rows.add(n=n, method=mt, k=k, seconds=dt, range_queries=rq,
                     clusters=int(lab.max()), finished=True)
            log(f"  fig5 n={n:,} {mt}: {dt:.3f}s, {rq:,} range queries")
        del tree, grid
    make_plots(out, log)


def fig5_full(args, xy, log, out):
    """Full vanilla DBSCAN at every size, smallest first. No time limit unless
    --baseline-hours; a stopped run is recorded as did-not-finish (DNF)."""
    eps, m = _eps(args, log), args.minpts
    rows = Rows(out / "fig5_scaling.csv", FIG5_FIELDS)
    (out / "labels").mkdir(exist_ok=True)
    for n in _sizes(args, len(xy)):
        if rows.has(n=n, method="vanilla_full"):
            continue
        sub = xy[:n]
        tree = IndexiconIndex(sub)
        state = {"t0": time.perf_counter(), "last": 0.0, "rq": 0, "el": 0.0}

        def progress(done, total, clusters, rq):
            el = time.perf_counter() - state["t0"]
            state.update(rq=rq, el=el, clusters=clusters)
            if done and el - state["last"] >= 60:
                state["last"] = el
                log(f"  vanilla full n={n:,}: {100 * done / total:5.1f}% of points, "
                    f"{rq:,} range queries, {hms(el)} elapsed, "
                    f"ETA ~{hms(el / max(rq, 1) * (total - rq))}, {rss_gb():.1f} GB")
            if args.baseline_hours and el > args.baseline_hours * 3600:
                raise _OutOfTime

        log(f"  vanilla full n={n:,}: started")
        try:
            lab, _, rq = DBSCAN_indexed(sub, eps, m, tree=tree, progress=progress)
        except (_OutOfTime, KeyboardInterrupt) as e:
            rows.add(n=n, method="vanilla_full", seconds=state["el"],
                     range_queries=state["rq"], finished=False, note=type(e).__name__)
            log(f"  vanilla full n={n:,}: DID NOT FINISH after {hms(state['el'])}")
            break                      # larger n would not finish either
        dt = time.perf_counter() - state["t0"]
        rows.add(n=n, method="vanilla_full", seconds=dt, range_queries=rq,
                 clusters=int(lab.max()), finished=True)
        log(f"  vanilla full n={n:,}: {hms(dt)}, {rq:,} range queries, "
            f"{int(lab.max()):,} clusters")
        np.save(out / "labels" / f"vanilla_full_n{n}.npy", lab.astype(np.int32))
        _add_hindsight_to_fig6(lab, n, out)
        make_plots(out, log)


# ---------------------------------------------------------------------------
# Figure 6: total points in the first k clusters vs. k
# ---------------------------------------------------------------------------
FIG6_FIELDS = ["n", "method", "k", "points_in_top_k", "cluster_size", "seconds"]


def _cluster_sizes(lab):
    """Sizes of clusters 1..C, in the order they were found."""
    if lab.max() < 1:
        return np.zeros(0, dtype=np.int64)
    return np.bincount(lab[lab > 0], minlength=int(lab.max()) + 1)[1:]


def fig6_experiment(args, xy, lat, lon, log, out):
    """One run per method with k = kmax. A run stopped at k finds exactly the
    first k clusters of this run (checked in testing), so one run gives the
    whole curve."""
    eps, m, K = _eps(args, log), args.minpts, args.fig6_kmax
    n = len(xy) if args.fig6_n in (0, None) or args.fig6_n > len(xy) else args.fig6_n
    rows = Rows(out / "fig6_topk_quality.csv", FIG6_FIELDS)
    methods = ["vanilla_topk", "grid_topk"] + ([] if args.no_maxrs else ["maxrs_topk"])
    todo = [mt for mt in methods if not rows.has(n=n, method=mt)]
    (out / "labels").mkdir(exist_ok=True)
    if todo:
        sub = xy[:n]
        tree, grid, _, _ = _build(sub, eps, log)
        for mt in todo:
            lab, dt, _ = _run(mt, sub, eps, m, K, tree, grid, args.fig_refine)
            np.save(out / "labels" / f"fig6_{mt}_n{n}.npy", lab.astype(np.int32))
            sizes = _cluster_sizes(lab)
            for kk, (s, c) in enumerate(zip(sizes, np.cumsum(sizes)), start=1):
                rows.add(n=n, method=mt, k=kk, points_in_top_k=int(c),
                         cluster_size=int(s), seconds=dt)
            log(f"  fig6 n={n:,} {mt}: {len(sizes)} clusters in {dt:.2f}s, "
                f"sizes {sizes[:10].tolist()}")
        del tree, grid
    full = out / "labels" / f"vanilla_full_n{n}.npy"
    if full.exists():
        _add_hindsight_to_fig6(np.load(full), n, out)
    make_plots(out, log)


def _add_hindsight_to_fig6(lab, n, out):
    """The k largest clusters of a full run: what a top-k could get at best,
    known only after clustering everything."""
    path = out / "fig6_topk_quality.csv"
    if not path.exists():
        return
    rows = Rows(path, FIG6_FIELDS)
    if not rows.has(n=n) or rows.has(n=n, method="vanilla_full_largest"):
        return
    kmax = max(int(r["k"]) for r in rows.rows if str(r["n"]) == str(n))
    sizes = np.sort(_cluster_sizes(lab))[::-1][:kmax]
    for kk, (s, c) in enumerate(zip(sizes, np.cumsum(sizes)), start=1):
        rows.add(n=n, method="vanilla_full_largest", k=kk,
                 points_in_top_k=int(c), cluster_size=int(s))


# ---------------------------------------------------------------------------
# Figure 7: the top-k clusters on a map, vanilla vs. grid
# ---------------------------------------------------------------------------
def fig7_experiment(args, xy, lat, lon, log, out):
    eps, m, k = _eps(args, log), args.minpts, args.fig_k
    n = len(xy) if args.fig7_n in (0, None) or args.fig7_n > len(xy) else args.fig7_n
    labels = {}
    for mt in ("vanilla_topk", "grid_topk"):
        f6 = out / "labels" / f"fig6_{mt}_n{n}.npy"
        if f6.exists() and args.fig6_kmax >= k:     # first k clusters of the fig6 run
            lab = np.load(f6).astype(np.int64)
            lab[lab > k] = -1
            labels[mt] = lab
    if len(labels) < 2:
        tree, grid, _, _ = _build(xy[:n], eps, log)
        for mt in ("vanilla_topk", "grid_topk"):
            labels[mt], dt, _ = _run(mt, xy[:n], eps, m, k, tree, grid, args.fig_refine)
            log(f"  fig7 {mt}: {dt:.2f}s")
        del tree, grid

    # Save only what the plot needs (a few MB), so it can be redrawn with
    # --plot-only without the 8.6M points.
    la, lo = lat[:n], lon[:n]
    rng = np.random.default_rng(0)
    bg = rng.choice(n, min(n, 300_000), replace=False)
    data = {"k": k, "n": n, "eps": eps, "minPts": m,
            "bg_lat": la[bg], "bg_lon": lo[bg]}
    rows = []
    for mt, lab in labels.items():
        for c in range(1, k + 1):
            idx = np.flatnonzero(lab == c)
            if len(idx) == 0:
                continue
            clat, clon = la[idx], lo[idx]
            pad = max(0.3 * max(np.ptp(clat), np.ptp(clon)), 0.01)
            box = (clon.min() - pad, clat.min() - pad, clon.max() + pad, clat.max() + pad)
            near = np.flatnonzero((lo >= box[0]) & (lo <= box[2]) &
                                  (la >= box[1]) & (la <= box[3]))
            data[f"{mt}_{c}_lat"], data[f"{mt}_{c}_lon"] = clat, clon
            data[f"{mt}_{c}_near_lat"], data[f"{mt}_{c}_near_lon"] = la[near], lo[near]
            rows.append([mt, c, len(idx), float(np.mean(clat)), float(np.mean(clon))])
    np.savez_compressed(out / "fig7_data.npz", **data)
    with open(out / "fig7_clusters.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "cluster", "size", "centroid_lat", "centroid_lon"])
        w.writerows(rows)
    log(f"  fig7 clusters: {rows}")
    make_plots(out, log)


# ---------------------------------------------------------------------------
# Table: Grid-DBSCAN top-k with each Indexicon index
# ---------------------------------------------------------------------------
TABLE_FIELDS = ["index", "n", "build_seconds", "grid_topk_seconds",
                "range_queries", "box_query_us", "index_mb"]


def _box_query_us(tree, xy, eps, n_q=100_000):
    """Index-only speed: mean microseconds per eps box query, over the same
    random points for every index. Grid-DBSCAN's run time also includes
    Python work per query that is the same for every index; this column
    isolates the part the index is responsible for."""
    pts = xy[np.random.default_rng(1).choice(len(xy), min(n_q, len(xy)), replace=False)]
    lo, hi = pts - eps, pts + eps
    q = tree.tree.query
    t0 = time.perf_counter()
    for i in range(len(pts)):
        q(lo[i], hi[i])
    return (time.perf_counter() - t0) / len(pts) * 1e6


def table_experiment(args, xy, lat, lon, log, out):
    eps, m, k = _eps(args, log), args.minpts, args.fig_k
    rows = Rows(out / "table_indexes.csv", TABLE_FIELDS)
    for n in _sizes(args, len(xy)):
        todo = [kind for kind in args.indexes if not rows.has(index=kind, n=n)]
        if not todo:
            continue
        sub = xy[:n]
        grid = HierarchicalGridIndex(sub, eps_min=eps, eps_max=eps)   # shared
        for kind in todo:
            t0 = time.perf_counter()
            tree = IndexiconIndex(sub, kind=kind)
            tb = time.perf_counter() - t0
            _, dt, rq = _timed("grid_topk", sub, eps, m, k, tree, grid,
                               args.fig_refine, args.repeats)
            us = _box_query_us(tree, sub, eps)
            rows.add(index=kind, n=n, build_seconds=tb, grid_topk_seconds=dt,
                     range_queries=rq, box_query_us=us,
                     index_mb=tree.nbytes() / 2**20)
            log(f"  table n={n:,} {kind}: build {tb:.2f}s, grid top-{k} {dt:.3f}s, "
                f"{rq:,} range queries, {us:.1f} us per box query")
            del tree
        del grid
    make_plots(out, log)


EXPERIMENTS = {"fig5": fig5_experiment, "fig6": fig6_experiment,
               "fig7": fig7_experiment, "table": table_experiment}


# ---------------------------------------------------------------------------
# Plotting (reads only the files in the results folder)
# ---------------------------------------------------------------------------
def make_plots(out, log):
    for name, fn in (("fig5", _plot_fig5), ("fig6", _plot_fig6),
                     ("fig7", _plot_fig7), ("table", _write_table)):
        try:
            fn(Path(out))
        except FileNotFoundError:
            pass
        except Exception as e:                   # a plot bug must not kill a long run
            log(f"  plotting {name} failed: {type(e).__name__}: {e}")


def _read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _save(fig, out, name):
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")


def _plot_fig5(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _read(out / "fig5_scaling.csv")
    k = next((r["k"] for r in rows if r["k"]), "?")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    all_n = sorted({int(r["n"]) for r in rows})
    for col, (ax, title) in enumerate(zip(
            axes, ["Runtime vs. dataset size", "Range queries vs. dataset size"])):
        for mt in ("vanilla_full", "vanilla_topk", "maxrs_topk", "grid_topk"):
            rs = sorted((r for r in rows if r["method"] == mt), key=lambda r: int(r["n"]))
            if not rs:
                continue
            st = STYLE[mt]
            label = st["label"].replace("stopped at k", f"stopped at k={k}")
            done = [r for r in rs if r["finished"] == "True"]
            x = [int(r["n"]) for r in done]
            y = [float(r["seconds"] if col == 0 else r["range_queries"]) for r in done]
            ax.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"],
                    label=label, lw=2, ms=7)
            for r in rs:
                if r["finished"] != "True":            # did not finish
                    yv = float(r["seconds"] if col == 0 else r["range_queries"])
                    ax.plot(int(r["n"]), yv, color=st["color"], marker="x", ms=10, mew=2)
                    ax.annotate("DNF", (int(r["n"]), yv), textcoords="offset points",
                                xytext=(6, 4), color=st["color"], fontsize=9)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(all_n)
        ax.set_xticklabels([_fmt_n(v) for v in all_n])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("Number of points (n)")
        ax.set_ylabel("Runtime (seconds)" if col == 0 else "Range queries issued")
        ax.set_title(title)
        ax.grid(True, which="major", alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Scaling, World-POI")
    fig.tight_layout()
    _save(fig, out, "fig5_scaling")
    plt.close(fig)


def _plot_fig6(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = _read(out / "fig6_topk_quality.csv")
    n = max(int(r["n"]) for r in rows)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mt in ("vanilla_full_largest", "vanilla_topk", "maxrs_topk", "grid_topk"):
        rs = sorted((r for r in rows if r["method"] == mt and int(r["n"]) == n),
                    key=lambda r: int(r["k"]))
        if not rs:
            continue
        st = STYLE[mt]
        ax.plot([int(r["k"]) for r in rs], [int(r["points_in_top_k"]) for r in rs],
                color=st["color"], marker=st["marker"], ls=st["ls"], label=st["label"],
                lw=2, ms=6)
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Total points in the first k clusters")
    ax.set_title(f"Top-k quality, World-POI ({_fmt_n(n)} points)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out, "fig6_topk_quality")
    plt.close(fig)


def _plot_fig7(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.load(out / "fig7_data.npz")
    k, n = int(z["k"]), int(z["n"])
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    methods = [("vanilla_topk", f"Vanilla DBSCAN + index (stopped at k={k})"),
               ("grid_topk", f"Grid-DBSCAN (k={k})")]
    fig = plt.figure(figsize=(4.2 + 2.6 * k, 7.5))
    gs = fig.add_gridspec(2, 1 + k, width_ratios=[3.2] + [1] * k, wspace=0.25, hspace=0.35)
    for row, (mt, title) in enumerate(methods):
        ax = fig.add_subplot(gs[row, 0])
        ax.scatter(z["bg_lon"], z["bg_lat"], s=0.05, c="#bbbbbb", rasterized=True, lw=0)
        sizes = []
        for c in range(1, k + 1):
            if f"{mt}_{c}_lat" not in z:
                continue
            clat, clon = z[f"{mt}_{c}_lat"], z[f"{mt}_{c}_lon"]
            col = colors[(c - 1) % len(colors)]
            sizes.append(len(clat))
            ax.scatter(np.mean(clon), np.mean(clat), s=220, facecolors="none",
                       edgecolors=col, lw=2, zorder=3)
            ax.annotate(str(c), (np.mean(clon), np.mean(clat)), xytext=(7, 7),
                        textcoords="offset points", color=col, fontweight="bold")
            zax = fig.add_subplot(gs[row, c])
            zax.scatter(z[f"{mt}_{c}_near_lon"], z[f"{mt}_{c}_near_lat"], s=1,
                        c="#cccccc", rasterized=True, lw=0)
            zax.scatter(clon, clat, s=3, c=col, rasterized=True, lw=0)
            zax.set_aspect(1 / np.cos(np.radians(np.mean(clat))))
            zax.set_title(f"#{c}: {len(clat):,} pts\n({np.mean(clat):.3f}, "
                          f"{np.mean(clon):.3f})", fontsize=8)
            zax.tick_params(labelsize=6)
            zax.ticklabel_format(useOffset=False, style="plain")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-60, 80)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\nsizes {sizes}", fontsize=10)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    fig.suptitle(f"Top-{k} clusters, World-POI ({_fmt_n(n)} points, eps = {float(z['eps']):g} km, "
                 f"minPts = {int(z['minPts'])}); grey = all points (sample on the world maps)",
                 fontsize=10)
    _save(fig, out, "fig7_topk_map")
    plt.close(fig)


def _write_table(out):
    rows = _read(out / "table_indexes.csv")
    kinds = list(dict.fromkeys(r["index"] for r in rows))
    ns = sorted({int(r["n"]) for r in rows})
    get = {(r["index"], int(r["n"])): r for r in rows}

    def block(field, fmt):
        best = {n: min((float(get[(kd, n)][field]) for kd in kinds if (kd, n) in get),
                       default=None) for n in ns}
        md = ["| Index | " + " | ".join(_fmt_n(n) for n in ns) + " |",
              "|---|" + "---:|" * len(ns)]
        tex = []
        for kd in kinds:
            cells_md, cells_tex = [], []
            for n in ns:
                if (kd, n) not in get:
                    cells_md.append("–"); cells_tex.append("--")
                    continue
                v = float(get[(kd, n)][field])
                s = fmt(v)
                is_best = best[n] is not None and v == best[n]
                cells_md.append(f"**{s}**" if is_best else s)
                cells_tex.append(f"\\textbf{{{s}}}" if is_best else s)
            md.append(f"| {kd} | " + " | ".join(cells_md) + " |")
            tex.append(kd.replace("_", "\\_") + " & " + " & ".join(cells_tex) + r" \\")
        head = ("Index & " + " & ".join(_fmt_n(n) for n in ns) + r" \\ \hline")
        return md, [r"\begin{tabular}{l" + "r" * len(ns) + "}", head] + tex + [r"\end{tabular}"]

    t = lambda v: f"{v:.3f}" if v < 10 else f"{v:.1f}"
    md1, tex1 = block("grid_topk_seconds", t)
    md2, tex2 = block("build_seconds", t)
    md3, tex3 = block("box_query_us", lambda v: f"{v:.1f}")
    text = ["# Grid-DBSCAN top-k runtime (s) per Indexicon index", "", *md1, "",
            "# Index-only: microseconds per eps box query", "", *md3, "",
            "# Index build time (s)", "", *md2, "", "Fastest per column in bold."]
    (out / "table_indexes.md").write_text("\n".join(text) + "\n")
    (out / "table_indexes.tex").write_text(
        "% Grid-DBSCAN top-k runtime (s)\n" + "\n".join(tex1) +
        "\n\n% microseconds per eps box query\n" + "\n".join(tex3) +
        "\n\n% index build time (s)\n" + "\n".join(tex2) + "\n")


if __name__ == "__main__":
    main()