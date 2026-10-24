"""
test_grid_index.py
===================
Checks, in order:
  1. the checker itself accepts valid labellings and rejects broken ones
  2. the baseline (DBSCAN_indexed_ids) is valid DBSCAN
  3. DBSCAN_grid_ids is valid DBSCAN on full runs
  4. ... and on truncated (top-k) runs
  5. one GridSeedIndex serves several minPts values
  6. refine > 1 is still valid
  7. points at identical coordinates are handled

Datasets include duplicate coordinates on purpose
"""

import numpy as np

from correctness import Reference, validate, assert_valid
from grid_index import GridSeedIndex
from spatial_index import RTreeIndex
try:  # once pasted into algorithms.py
    from algorithms import DBSCAN_indexed_ids, DBSCAN_grid_ids
except ImportError:
    from algorithms_additions import DBSCAN_indexed_ids, DBSCAN_grid_ids


def make_data(n, n_clusters, seed, noise=0.7, dup_frac=0.05, buildings=5):
    """Gaussian clusters on uniform noise, with duplicate coordinates."""
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / 10.0)
    per = int(n * (1 - noise)) // max(n_clusters, 1)
    parts = []
    for _ in range(n_clusters):
        c = rng.uniform(0.15 * side, 0.85 * side, 2)
        parts.append(rng.normal(c, 0.12 * rng.uniform(0.6, 1.6), size=(per, 2)))
    parts.append(rng.uniform(0, side, size=(n - per * n_clusters, 2)))
    xy = np.vstack(parts)
    # copy some points onto existing locations
    n_dup = int(len(xy) * dup_frac)
    xy[rng.choice(len(xy), n_dup, replace=False)] = xy[rng.choice(len(xy), n_dup)]
    # a few locations with several points stacked on them
    for _ in range(buildings):
        spot = rng.uniform(0, side, 2)
        k = int(rng.integers(3, 12))
        xy = np.vstack([xy, np.repeat(spot[None, :], k, axis=0)])
    return xy[rng.permutation(len(xy))]


CASES = [  # (n, clusters, eps, minPts, seed)
    (600, 3, 0.30, 5, 1), (1500, 5, 0.30, 5, 2), (1500, 5, 0.20, 4, 3),
    (2500, 6, 0.30, 8, 4), (2500, 6, 0.45, 5, 5), (4000, 8, 0.25, 6, 6),
    (800, 0, 0.30, 5, 7),   # noise only
    (800, 2, 3.00, 5, 8),   # eps so large almost everything joins
    (300, 1, 0.30, 3, 9),
]

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"  FAIL {name} {detail}")


def main():
    # 1. the checker rejects broken labellings
    print("1. checker sanity")
    xy = make_data(1500, 5, seed=11)
    eps, mp = 0.30, 5
    ref = Reference(xy, eps, mp)
    good, _, _ = DBSCAN_indexed_ids(xy, eps, mp)
    check("accepts a valid labelling", validate(good, xy, eps, mp, ref=ref)[0])

    ids = [L for L in np.unique(good) if L != -1]
    core = np.flatnonzero(ref.core)

    bad = good.copy(); bad[bad == ids[1]] = ids[0]
    check("rejects two clusters merged", not validate(bad, xy, eps, mp, ref=ref)[0])

    bad = good.copy(); members = np.flatnonzero((bad == ids[0]) & ref.core)
    bad[members[: len(members) // 2]] = 999
    check("rejects a cluster split in two", not validate(bad, xy, eps, mp, ref=ref)[0])

    bad = good.copy(); bad[core[0]] = -1
    check("rejects a core point marked noise", not validate(bad, xy, eps, mp, ref=ref)[0])

    bad = good.copy(); noise = np.flatnonzero((good == -1) & ~ref.core)
    bad[noise[0]] = ids[0]
    check("rejects noise put into a cluster", not validate(bad, xy, eps, mp, ref=ref)[0])

    bad = good.copy(); bad[bad == ids[-1]] = -1
    check("rejects a missing cluster (full run)", not validate(bad, xy, eps, mp, ref=ref)[0])
    check("accepts the same as a truncated run",
          validate(bad, xy, eps, mp, k=len(ids) - 1, ref=ref)[0])

    # 2-4. baseline and grid, full and truncated 
    print("2-4. baseline and grid on", len(CASES), "datasets, full and truncated")
    for n, ncl, eps, mp, seed in CASES:
        xy = make_data(n, ncl, seed)
        ref = Reference(xy, eps, mp)
        rt = RTreeIndex(xy)
        idx = GridSeedIndex(xy, eps)
        tag = f"(n={len(xy)} eps={eps} minPts={mp})"

        lab, _, _ = DBSCAN_indexed_ids(xy, eps, mp, rtree=rt)
        ok, probs, _ = validate(lab, xy, eps, mp, ref=ref)
        check(f"baseline full {tag}", ok, probs[:2])

        lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, rtree=rt, index=idx)
        ok, probs, _ = validate(lab, xy, eps, mp, ref=ref)
        check(f"grid full {tag}", ok, probs[:2])

        for k in (1, 3, 5):
            lab, _, _ = DBSCAN_indexed_ids(xy, eps, mp, max_iterations=k, rtree=rt)
            ok, probs, _ = validate(lab, xy, eps, mp, k=k, ref=ref)
            check(f"baseline k={k} {tag}", ok, probs[:2])
            lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, max_iterations=k,
                                           rtree=rt, index=idx)
            ok, probs, _ = validate(lab, xy, eps, mp, k=k, ref=ref)
            check(f"grid k={k} {tag}", ok, probs[:2])

    # 5. one index, many minPts 
    print("5. one index reused across minPts")
    xy = make_data(2500, 6, seed=21)
    eps = 0.30
    rt = RTreeIndex(xy)
    idx = GridSeedIndex(xy, eps)            # built once
    for mp in (3, 5, 8, 12, 20):
        lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, rtree=rt, index=idx)
        ok, probs, _ = validate(lab, xy, eps, mp)
        check(f"minPts={mp} on shared index", ok, probs[:2])

    # 6. refine 
    print("6. refine")
    xy = make_data(2500, 6, seed=31)
    eps, mp = 0.30, 5
    ref = Reference(xy, eps, mp)
    rt = RTreeIndex(xy)
    idx = GridSeedIndex(xy, eps)
    for m in (2, 4, 8):
        for k in (None, 3):
            lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, max_iterations=k,
                                           rtree=rt, index=idx, refine=m)
            ok, probs, _ = validate(lab, xy, eps, mp, k=k, ref=ref)
            check(f"refine={m} k={k}", ok, probs[:2])

    # 7. stacked duplicates 
    print("7. identical coordinates")
    # five points at one address: core under DBSCAN, so it must be a cluster
    xy = np.array([(3.0, 3.0)] * 5 + [(20.0, 20.0), (40.0, 5.0)])
    lab, _, _, _ = DBSCAN_grid_ids(xy, 1.0, 5)
    check("5 points at one address form a cluster",
          validate(lab, xy, 1.0, 5)[0] and len(set(lab[:5])) == 1 and lab[0] != -1,
          f"labels={lab.tolist()}")
    # four at one address is one short
    xy = np.array([(3.0, 3.0)] * 4 + [(20.0, 20.0)])
    lab, _, _, _ = DBSCAN_grid_ids(xy, 1.0, 5)
    check("4 points at one address are noise", (lab == -1).all(), f"labels={lab.tolist()}")

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()