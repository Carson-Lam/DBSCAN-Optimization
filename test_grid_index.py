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
  8. hierarchy: rung selection picks the right rung for every eps
  9. hierarchy: the rungs are real nested grids
 10. hierarchy: valid DBSCAN at many eps, on and between rungs
 11. hierarchy: one index serves many (eps, minPts) pairs
 12. hierarchy: mistakes are refused, not silently wrong

Datasets include duplicate coordinates on purpose
"""

import numpy as np

from correctness import Reference, validate
from grid_index import GridSeedIndex, HierarchicalGridIndex, SQRT2
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

    # 8. rung selection
    print("8. hierarchy: rung selection")
    rng = np.random.default_rng(41)
    xy = make_data(1500, 5, seed=41)
    H = HierarchicalGridIndex(xy, eps_min=0.05, eps_max=1.6)
    sides = H.sides
    check("ratio between rungs is exactly 2", np.allclose(sides[1:] / sides[:-1], 2.0))
    for eps in np.concatenate([rng.uniform(0.05, 1.6, 300),
                               [0.05, 1.6], sides[(sides >= 0.05) & (sides <= 1.6)]]):
        f, c = H.fine_rung(eps), H.coarse_rung(eps)
        bad = []
        if f is None or c is None:
            bad.append("no rung")
        else:
            if not sides[f] * SQRT2 <= eps:
                bad.append(f"fine diagonal {sides[f] * SQRT2:.6g} > eps")
            if f + 1 < len(sides) and sides[f + 1] * SQRT2 <= eps * (1 - 1e-6):
                bad.append("a larger rung would also have been safe for fine")
            if not sides[c] >= eps:
                bad.append(f"coarse side {sides[c]:.6g} < eps")
            if c > 0 and sides[c - 1] >= eps * (1 + 1e-6):
                bad.append("a smaller rung would also have been safe for coarse")
            if not (sides[f] > eps / (2 * SQRT2) * (1 - 1e-6)):
                bad.append("fine rung more than 2x too small")
            if not (sides[c] < 2 * eps * (1 + 1e-6)):
                bad.append("coarse rung more than 2x too big")
        check(f"rungs for eps={eps:.6g}", not bad, bad)
    # the meeting-notes example: the rule must use eps/sqrt(2), not eps
    H2 = HierarchicalGridIndex(xy, eps_min=0.0707, eps_max=1.0)
    f = H2.fine_rung(0.1)
    check("fine rung for eps=0.1 has diagonal <= 0.1",
          H2.sides[f] * SQRT2 <= 0.1, f"side={H2.sides[f]:.4f}")

    # 9. the rungs are nested grids
    print("9. hierarchy: rungs are nested grids")
    xy = make_data(3000, 6, seed=43)
    H = HierarchicalGridIndex(xy, eps_min=0.04, eps_max=2.0)
    po = H.point_order
    for lv in H.levels:
        # every cell's points really lie in one square of that rung's side
        naive = np.floor((xy - H.origin) / lv.side).astype(np.int64)
        member_ok = True
        for c in range(lv.n_cells):
            ids = po[lv.starts[c]: lv.starts[c] + lv.counts[c]]
            if len(np.unique(naive[ids], axis=0)) != 1:
                member_ok = False
                break
        check(f"rung {lv.rung}: each cell is one square", member_ok)
        # counts match a grid built from scratch at that side
        _, naive_counts = np.unique(naive, axis=0, return_counts=True)
        check(f"rung {lv.rung}: counts match a fresh grid",
              np.array_equal(np.sort(naive_counts), np.sort(lv.counts)))
        check(f"rung {lv.rung}: counts sum to n", int(lv.counts.sum()) == len(xy))
        # 3x3 totals match a direct count (only rungs usable as a coarse grid
        # have them)
        check(f"rung {lv.rung}: has the arrays its role needs",
              (lv.block is not None) == lv.as_coarse
              and (lv.by_count is not None) == lv.as_fine)
        if lv.block is None:
            continue
        cells = {tuple(k): v for k, v in zip(*np.unique(naive, axis=0, return_counts=True))}
        sample = rng.choice(lv.n_cells, min(200, lv.n_cells), replace=False)
        blk_ok = True
        for c in sample:
            i, j = naive[po[lv.starts[c]]]
            direct = sum(cells.get((i + di, j + dj), 0)
                         for di in (-1, 0, 1) for dj in (-1, 0, 1))
            if direct != lv.block[c]:
                blk_ok = False
                break
        check(f"rung {lv.rung}: 3x3 totals match a direct count", blk_ok)
    # each cell is exactly 4 cells (or fewer, if empty) one rung down
    nest_ok = True
    for lo, hi in zip(H.levels[:-1], H.levels[1:]):
        parent = lo.cell_keys >> np.uint64(2)
        pos = np.searchsorted(hi.cell_keys, parent)
        if not np.array_equal(hi.cell_keys[pos], parent):
            nest_ok = False
        summed = np.bincount(pos, weights=lo.counts, minlength=hi.n_cells)
        if not np.array_equal(summed.astype(np.int64), hi.counts.astype(np.int64)):
            nest_ok = False
    check("each cell's children sum to its count", nest_ok)

    # 10. DBSCAN on the hierarchy, many eps
    print("10. hierarchy: valid DBSCAN across eps")
    for n, ncl, mp, seed in [(1500, 5, 5, 51), (2500, 6, 8, 52), (800, 0, 5, 53)]:
        xy = make_data(n, ncl, seed)
        rt = RTreeIndex(xy)
        H = HierarchicalGridIndex(xy, eps_min=0.12, eps_max=0.9)
        on_rung = [s for s in H.sides if 0.12 <= s <= 0.9]
        eps_list = sorted(set(np.round(np.geomspace(0.12, 0.9, 9), 6).tolist() + on_rung))
        for eps in eps_list:
            ref = Reference(xy, eps, mp)
            for k in (None, 1, 3):
                for refine in (1, 4):
                    lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, max_iterations=k,
                                                   rtree=rt, index=H, refine=refine)
                    ok, probs, _ = validate(lab, xy, eps, mp, k=k, ref=ref)
                    check(f"hier n={len(xy)} eps={eps:.4g} mp={mp} k={k} refine={refine}",
                          ok, probs[:2])

    # 11. one index, many (eps, minPts)
    print("11. hierarchy: one index, many (eps, minPts)")
    xy = make_data(2500, 6, seed=61)
    rt = RTreeIndex(xy)
    H = HierarchicalGridIndex(xy, eps_min=0.1, eps_max=1.0)   # built once
    for eps in (0.1, 0.17, 0.3, 0.55, 1.0):
        for mp in (3, 6, 15):
            lab, _, _, _ = DBSCAN_grid_ids(xy, eps, mp, rtree=rt, index=H)
            ok, probs, _ = validate(lab, xy, eps, mp)
            check(f"shared index eps={eps} minPts={mp}", ok, probs[:2])
    # matches the single-eps index on the same query (same clusters)
    for eps in (0.17, 0.3):
        a, _, _, _ = DBSCAN_grid_ids(xy, eps, 5, rtree=rt, index=H)
        b, _, _, _ = DBSCAN_grid_ids(xy, eps, 5, rtree=rt, index=GridSeedIndex(xy, eps))
        ref = Reference(xy, eps, 5)
        same = validate(a, xy, eps, 5, ref=ref)[0] and validate(b, xy, eps, 5, ref=ref)[0] \
            and np.array_equal(a == -1, b == -1)
        check(f"hierarchy and single-eps index agree on noise, eps={eps}", same)

    # 12. refused, not silently wrong
    print("12. hierarchy: bad requests are refused")
    xy = make_data(600, 3, seed=71)
    H = HierarchicalGridIndex(xy, eps_min=0.2, eps_max=0.6)
    lo, hi = H.eps_range
    check("eps_range covers what was asked for", lo <= 0.2 and hi >= 0.6, (lo, hi))
    for ok_eps in (0.2, 0.6, hi):
        try:
            DBSCAN_grid_ids(xy, ok_eps, 5, index=H)
            check(f"eps={ok_eps:.6g} inside range accepted", True)
        except ValueError as e:
            check(f"eps={ok_eps:.6g} inside range accepted", False, str(e))
    for bad_eps in (0.1, 0.19, hi * 1.001, 2 * hi):
        try:
            DBSCAN_grid_ids(xy, bad_eps, 5, index=H)
            check(f"eps={bad_eps} outside range refused", False)
        except ValueError:
            check(f"eps={bad_eps} outside range refused", True)
    try:
        DBSCAN_grid_ids(xy, 0.3, 5, index=GridSeedIndex(xy, 0.4))
        check("single-eps index refuses another eps", False)
    except ValueError:
        check("single-eps index refuses another eps", True)
    try:
        DBSCAN_grid_ids(xy[:500], 0.3, 5, index=H)
        check("index over different points refused", False)
    except ValueError:
        check("index over different points refused", True)
    for args in [(0.0, 1.0), (1.0, 0.5), (-1.0, 1.0)]:
        try:
            HierarchicalGridIndex(xy, *args)
            check(f"eps range {args} refused", False)
        except ValueError:
            check(f"eps range {args} refused", True)
    # stacked duplicates through the hierarchy
    xy = np.array([(3.0, 3.0)] * 5 + [(20.0, 20.0), (40.0, 5.0)])
    H = HierarchicalGridIndex(xy, eps_min=0.5, eps_max=2.0)
    lab, _, _, _ = DBSCAN_grid_ids(xy, 1.0, 5, index=H)
    check("hierarchy: 5 points at one address form a cluster",
          lab[0] != -1 and len(set(lab[:5])) == 1, f"labels={lab.tolist()}")

    print(f"\n{passed} passed, {failed} failed")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()