"""
bench_hgrid.py
===============
Measurements for the hierarchical grid. Run: python bench_hgrid.py [part]

  cost     index build time and memory vs n: single-eps vs hierarchy

  price    the price of reuse: hierarchy vs an index built for exactly
           that eps, as eps moves across one octave between rungs.
           Reports how far each rung is from the ideal size, range
           queries, lemon seeds and top-k fidelity.

  reuse    one hierarchy build + many (eps, minPts) queries, vs building
           a single-eps index for every query

  noise    when does phase 2 stop early? Lemon seeds (noise points
           validated one by one) as noise density rises, single vs
           hierarchy. The coarse proof only works where a 3x3 block of
           noise holds fewer than minPts points.

Writes CSVs to results/. Only synthetic data so far since the real-data runs go in
experiments.py after the dataset is settled.

Top-k fidelity here means: the k clusters returned are the k clusters
with the highest PEAK density (largest eps-neighbourhood of any of their
points). k=1 is therefore also an ordering check. 
"""

import csv
import os
import sys
import time

import numpy as np

from algorithms import DBSCAN_grid_ids
from correctness import Reference
from grid_index import GridSeedIndex, HierarchicalGridIndex, SQRT2
from spatial_index import RTreeIndex

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)


def save(rows, header, name):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"  saved {path}")


def varied_clusters(n, n_clusters, seed, noise=0.7, spread=0.12):
    """Gaussian clusters of varying tightness on uniform noise,
    10 points per unit^2 overall."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / 10.0)
    per = int(n * (1 - noise)) // n_clusters
    parts = [rng.normal(rng.uniform(0.15 * side, 0.85 * side, 2),
                        spread * rng.uniform(0.6, 1.6), size=(per, 2))
             for _ in range(n_clusters)]
    parts.append(rng.uniform(0, side, size=(n - per * n_clusters, 2)))
    xy = np.clip(np.vstack(parts), 0, side)
    return xy[rng.permutation(len(xy))]


def fixed_clusters(n, seed=1, n_clusters=5, per=300):
    """Cluster content fixed; n grows by adding noise only."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / 10.0)
    c = rng.uniform(0.15 * side, 0.85 * side, size=(n_clusters, 2))
    parts = [rng.normal(ci, 0.12, size=(per, 2)) for ci in c]
    parts.append(rng.uniform(0, side, size=(n - per * n_clusters, 2)))
    xy = np.clip(np.vstack(parts), 0, side)
    return xy[rng.permutation(len(xy))]


def peak_density(ref):
    """Per reference cluster: the largest eps-neighbourhood among its points."""
    out = np.zeros(ref.n_comp)
    core = np.flatnonzero(ref.core)
    np.maximum.at(out, ref.comp[core], ref.nbr_count[core])
    return out


def topk_hits(labels, ref, k):
    """How many of the k returned clusters are among the true k densest."""
    truth = set(np.argsort(-peak_density(ref), kind="stable")[:k].tolist())
    got = set()
    core = ref.core
    for L in np.unique(labels[labels != -1]):
        comps = np.unique(ref.comp[(labels == L) & core])
        if len(comps) == 1:
            got.add(int(comps[0]))
    return len(got & truth)


def part_cost():
    print("\n[cost] index build time and memory")
    rows = []
    for n in (100_000, 1_000_000, 5_000_000):
        rng = np.random.default_rng(0)
        xy = rng.uniform(0, np.sqrt(n / 10.0), size=(n, 2))
        t0 = time.time(); s = GridSeedIndex(xy, 0.30); ts = time.time() - t0
        t0 = time.time(); h = HierarchicalGridIndex(xy, 0.05, 1.6); th = time.time() - t0
        lo, hi = h.eps_range
        rows.append([n, ts, s.nbytes(), th, h.nbytes(), len(h.sides), lo, hi])
        print(f"  n={n:>10,}  single {ts:6.2f}s {s.nbytes()/2**20:7.1f} MB | "
              f"hierarchy {th:6.2f}s {h.nbytes()/2**20:7.1f} MB, "
              f"{len(h.sides)} rungs, eps {lo:.3g}..{hi:.3g}")
    save(rows, ["n", "single_build_s", "single_bytes", "hier_build_s",
                "hier_bytes", "rungs", "eps_lo", "eps_hi"], "h1_index_cost.csv")


def part_price(trials=30, n=3000, minPts=5):
    print(f"\n[price] hierarchy vs exact single-eps index, {trials} trials, n={n}")
    # one octave of eps, from one coarse rung side to the next
    lo_eps = 0.25
    eps_list = lo_eps * 2.0 ** np.linspace(0, 1, 6)
    acc = {}  # (eps, index, refine) -> dict of sums
    for t in range(trials):
        xy = varied_clusters(n, 6, seed=1000 + t)
        rt = RTreeIndex(xy)
        H = HierarchicalGridIndex(xy, eps_min=0.1, eps_max=1.0)
        for eps in eps_list:
            ref = Reference(xy, eps, minPts)
            if ref.n_comp < 3:
                continue
            S = GridSeedIndex(xy, eps)
            for name, idx in (("hierarchy", H), ("single", S)):
                for refine in (1, 8):
                    a = acc.setdefault((eps, name, refine), dict(
                        trials=0, rq_full=0, lemons=0, rq_k3=0, hit1=0, hit3=0))
                    lab, outer, _, rq = DBSCAN_grid_ids(
                        xy, eps, minPts, rtree=rt, index=idx, refine=refine)
                    a["trials"] += 1
                    a["rq_full"] += rq
                    a["lemons"] += outer - len(np.unique(lab[lab != -1]))
                    lab1, *_ = DBSCAN_grid_ids(xy, eps, minPts, max_iterations=1,
                                               rtree=rt, index=idx, refine=refine)
                    lab3, _, _, rq3 = DBSCAN_grid_ids(xy, eps, minPts, max_iterations=3,
                                                      rtree=rt, index=idx, refine=refine)
                    a["rq_k3"] += rq3
                    a["hit1"] += topk_hits(lab1, ref, 1)
                    a["hit3"] += topk_hits(lab3, ref, 3) / 3
        print(f"  trial {t + 1}/{trials}", end="\r", flush=True)
    print()

    probe = HierarchicalGridIndex(varied_clusters(n, 6, seed=1000), 0.1, 1.0)
    rows = []
    print(f"  {'eps':>6} {'fine/ideal':>10} {'coarse/ideal':>12} {'index':>9} {'ref':>3} "
          f"{'rq full':>8} {'lemons':>7} {'rq k=3':>7} {'k=1 hit':>8} {'k=3 frac':>9}")
    for eps in eps_list:
        f, c = probe.fine_rung(eps), probe.coarse_rung(eps)
        fr = probe.sides[f] / (eps / SQRT2)
        cr = probe.sides[c] / eps
        for name in ("single", "hierarchy"):
            for refine in (1, 8):
                a = acc.get((eps, name, refine))
                if not a:
                    continue
                m = a["trials"]
                fine_ratio = 1.0 if name == "single" else fr
                coarse_ratio = 1.0 if name == "single" else cr
                row = [eps, name, refine, fine_ratio, coarse_ratio, m,
                       a["rq_full"] / m, a["lemons"] / m, a["rq_k3"] / m,
                       a["hit1"] / m, a["hit3"] / m]
                rows.append(row)
                print(f"  {eps:6.3f} {fine_ratio:10.2f} {coarse_ratio:12.2f} {name:>9} "
                      f"{refine:>3} {row[6]:8.0f} {row[7]:7.1f} {row[8]:7.0f} "
                      f"{row[9]:8.2f} {row[10]:9.2f}")
    save(rows, ["eps", "index", "refine", "fine_side_over_ideal",
                "coarse_side_over_ideal", "trials", "rq_full", "lemon_seeds",
                "rq_k3", "k1_hit_rate", "k3_fraction_recovered"],
         "h2_price_of_reuse.csv")


def part_reuse(n=40_000, k=3):
    print(f"\n[reuse] one hierarchy vs a fresh single-eps index per query, n={n}")
    xy = fixed_clusters(n)
    rt = RTreeIndex(xy)
    queries = [(eps, mp) for eps in (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
               for mp in (4, 6, 10)]

    t0 = time.time()
    H = HierarchicalGridIndex(xy, eps_min=0.2, eps_max=0.5)
    t_build_h = time.time() - t0
    t0 = time.time()
    rq_h = sum(DBSCAN_grid_ids(xy, e, m, max_iterations=k, rtree=rt, index=H)[3]
               for e, m in queries)
    t_query_h = time.time() - t0

    t0 = time.time()
    rq_s = 0
    for e, m in queries:
        rq_s += DBSCAN_grid_ids(xy, e, m, max_iterations=k, rtree=rt,
                                index=GridSeedIndex(xy, e))[3]
    t_single = time.time() - t0

    print(f"  {len(queries)} queries (6 eps x 3 minPts), top-{k}")
    print(f"  hierarchy: build {t_build_h:.2f}s once + queries {t_query_h:.2f}s "
          f"= {t_build_h + t_query_h:.2f}s, {rq_h} range queries")
    print(f"  single   : build+query each time {t_single:.2f}s, {rq_s} range queries")
    save([[len(queries), t_build_h, t_query_h, rq_h, t_single, rq_s]],
         ["queries", "hier_build_s", "hier_query_s", "hier_rq",
          "single_total_s", "single_rq"], "h3_reuse.csv")


def part_noise(n_noise=20_000, eps=0.3, minPts=5, k=3):
    print(f"\n[noise] lemon seeds vs noise density, eps={eps}, minPts={minPts}")
    print("  expected noise points in an eps-disc:  pi*eps^2*rho")
    print("  expected noise points in a 3x3 block:  9*s^2*rho  (s = coarse side)")
    rows = []
    for rho in (0.5, 1, 2, 3, 5, 10):
        side = np.sqrt(n_noise / rho)
        rng = np.random.default_rng(7)
        c = rng.uniform(0.15 * side, 0.85 * side, size=(5, 2))
        xy = np.vstack([rng.normal(ci, 0.12, size=(300, 2)) for ci in c]
                       + [rng.uniform(0, side, size=(n_noise, 2))])
        rt = RTreeIndex(xy)
        # range chosen so eps falls just above a rung: coarse rung ~1.9x eps,
        # fine rung ~0.67x ideal close to the hierarchy's worst case
        H = HierarchicalGridIndex(xy, eps_min=0.672 * eps, eps_max=eps)
        out = [rho, np.pi * eps ** 2 * rho]
        for name, idx in (("single", GridSeedIndex(xy, eps)), ("hierarchy", H)):
            _, coarse = idx.levels_for(eps)
            lab, outer, _, rq = DBSCAN_grid_ids(xy, eps, minPts, rtree=rt, index=idx)
            lemons = outer - len(np.unique(lab[lab != -1]))
            _, _, _, rqk = DBSCAN_grid_ids(xy, eps, minPts, max_iterations=k,
                                           rtree=rt, index=idx)
            out += [coarse.side / eps, 9 * coarse.side ** 2 * rho, lemons, rq, rqk]
        rows.append(out)
        print(f"  rho={rho:>4}  disc {out[1]:4.2f} | single: block {out[3]:5.2f} "
              f"lemons {out[4]:>6} rq_full {out[5]:>6} rq_k{k} {out[6]:>5} | "
              f"hier (coarse {out[7]:.2f}x): block {out[8]:5.2f} lemons {out[9]:>6} "
              f"rq_full {out[10]:>6} rq_k{k} {out[11]:>5}")
    save(rows, ["noise_density", "expected_in_disc",
                "single_coarse_over_eps", "single_expected_block", "single_lemons",
                "single_rq_full", f"single_rq_k{k}",
                "hier_coarse_over_eps", "hier_expected_block", "hier_lemons",
                "hier_rq_full", f"hier_rq_k{k}"], "h4_noise_density.csv")


if __name__ == "__main__":
    parts = sys.argv[1:] or ["cost", "price", "reuse", "noise"]
    for p in parts:
        {"cost": part_cost, "price": part_price, "reuse": part_reuse,
         "noise": part_noise}[p]()