"""
algorithms.py
=============
The DBSCAN variants compared in the paper.

  DBSCAN_indexed       Baseline: ordinary DBSCAN, every range query through
                       a spatial tree. Visits points in input order.

  DBSCAN_grid          Our method: the same DBSCAN, but each cluster is
                       started from the densest remaining grid cell, so
                       clusters come out densest-first and a top-k run can
                       stop after k.

  DBSCAN_single_maxrs  Single-run MaxRS: each point's best eps/sqrt(2)
                       square is computed once, and clusters are seeded
                       from the densest squares first.

All three identify points by their position in the input array, so points
at identical coordinates stay separate, and return labels as an array:
-1 = noise, 1..C = cluster id.

Two structures can be passed in, built once and reused:

  tree   answers every range query ("which points are within eps of point
         i?"). An IndexiconIndex R-tree by default. 

  grid   picks where each cluster starts (DBSCAN_grid only). A
         HierarchicalGridIndex serves any eps in its range; a GridSeedIndex
         serves one eps.

Usage:
    labels, *_ = DBSCAN_indexed(xy, eps, minPts, tree=tree)                   
    labels, *_ = DBSCAN_grid(xy, eps, minPts, max_iterations=k, tree=tree, grid=grid)

"""

import time

import numpy as np

from spatial_index import IndexiconIndex
from grid_index import GridSeedIndex, _Walk, fine_seed, fine_candidates, coarse_seed

_UNSEEN = -2   # not visited yet; turned into -1 (noise) before returning


# ---------------------------------------------------------------------------
# Single-run MaxRS DBSCAN
# ---------------------------------------------------------------------------
def _phase1_squares(xy, side, tree):
    """For each point p: (count, bottom y) of the best side x side square
    with left edge at p.x that contains p. Static: counts all points."""
    n = len(xy)
    count = np.zeros(n, dtype=np.int32)
    bottom = np.empty(n, dtype=np.float64)
    q = tree.tree.query
    lo = np.empty(2)
    hi = np.empty(2)
    X, Y = xy[:, 0], xy[:, 1]
    pad = 1e-12 * (np.abs(xy).max() + side)
    for i in range(n):
        x, y = X[i], Y[i]
        lo[0] = x - pad; hi[0] = x + side + pad
        lo[1] = y - side - pad; hi[1] = y + side + pad
        ids = q(lo, hi)
        xi = X[ids]
        ys = np.sort(Y[ids[(xi >= x) & (xi <= x + side)]])   # drop the pad
        # candidate bottoms: point heights between y - side and y
        a = np.searchsorted(ys, y - side, "left")
        b = np.searchsorted(ys, y, "right")
        bots = ys[a:b]
        c = np.searchsorted(ys, bots + side, "right") - np.arange(a, b)
        j = int(np.argmax(c))
        count[i] = c[j]
        bottom[i] = bots[j]
    return count, bottom


def _phase2_counts(xy, eps, tree):
    """For each point: number of points in the 2eps square centred on it."""
    n = len(xy)
    out = np.zeros(n, dtype=np.int32)
    q = tree.tree.query
    lo = np.empty(2)
    hi = np.empty(2)
    pad = 1e-12 * (np.abs(xy).max() + eps)
    r = eps + pad
    for i in range(n):
        x, y = xy[i]
        lo[0] = x - r; lo[1] = y - r
        hi[0] = x + r; hi[1] = y + r
        out[i] = len(q(lo, hi))
    return out


def DBSCAN_single_maxrs(DB, eps, minPts, max_iterations=None, tree=None,
                        timings=None):
    """Single-run MaxRS DBSCAN (Lance's method, rewritten to run on the same
    points and the same tree as DBSCAN_indexed and DBSCAN_grid).

    Changes from Lance's first version, so it runs and stays exact:

      * Points are identified by index (position in the array), not by
        coordinate tuples, so duplicate coordinates stay separate points.
      * Range queries go through the shared IndexiconIndex.
      * Counts are computed once over ALL points (static), never only over the
        points still unlabeled. Counting only unlabeled points can make a core
        point look like noise and end the run too early.

    Phase 1 (seeds that are certainly core)
        For every point p, the "single run" MaxRS: the best square of side
        eps/sqrt(2) whose left edge is at p.x and which contains p (bottom edge
        anywhere between p.y - side and p.y). Its count is stored per point,
        once, and the points are visited from the highest count down. Any square
        of that side has diagonal eps, so if it holds >= minPts points, every
        point in it is core: seed from any unlabeled one and expand. Stop when
        the next square holds < minPts points.

    Phase 2 (candidates, needs a check)
        For every point p, the number of points in the 2eps x 2eps square
        centred on p. That square contains p's eps-circle, so the count is an
        upper bound on p's neighbourhood. Points are visited from the highest
        count down and range-queried; core ones start a cluster. When the next
        count is < minPts, no core point is left, so the run can stop.

        (Lance's version picks the best 2eps square per point here too. That
        square need not contain the core point that justified it, so a core
        point could be skipped. The centred square cannot miss one.)

    Range queries counted in the returned total are eps-range queries only, the
    same as in the other two algorithms. The square counts are box queries on
    the same tree and show up in the run time, not in the range-query count.

    Parameters as DBSCAN_grid. timings: optional dict; filled with the
    seconds spent building the phase-1 and phase-2 square counts.

    Returns (labels, outer_iterations, maxrs_runs, range_queries).
    labels: -1 noise, 1..C cluster id, in the order clusters were found.
    """
    xy = np.ascontiguousarray(DB, dtype=float)
    n = len(xy)
    if tree is None:
        tree = IndexiconIndex(xy)
    if tree.n != n:
        raise ValueError(f"tree was built over {tree.n} points, DB has {n}")
    neighbors_of = tree.neighbors_of
    side = eps / np.sqrt(2.0)

    labels = [_UNSEEN] * n
    C = 0
    outer = 0
    maxrs_runs = 0
    range_queries = 0

    def budget_left():
        return max_iterations is None or C < max_iterations

    def expand(seed, N, cid):
        nonlocal range_queries
        labels[seed] = cid
        S = set(N)
        S.discard(seed)
        while S:
            q = S.pop()
            if labels[q] == -1:
                labels[q] = cid
            if labels[q] != _UNSEEN:
                continue
            labels[q] = cid
            Nq = neighbors_of(q, eps)
            range_queries += 1
            if len(Nq) >= minPts:
                S.update(Nq)

    # Phase 1
    if budget_left():
        t0 = time.time()
        count, bottom = _phase1_squares(xy, side, tree)
        maxrs_runs += 1
        if timings is not None:
            timings["phase1_squares_s"] = time.time() - t0
        order = np.argsort(-count, kind="stable")
        lo = np.empty(2)
        hi = np.empty(2)
        for a in order:
            if not budget_left() or count[a] < minPts:
                break
            lo[0] = xy[a, 0]; hi[0] = xy[a, 0] + side
            lo[1] = bottom[a]; hi[1] = bottom[a] + side
            live = [int(i) for i in tree.tree.query(lo, hi) if labels[i] == _UNSEEN]
            if not live:
                continue
            outer += 1
            seed = live[0]
            N = neighbors_of(seed, eps)
            range_queries += 1
            if len(N) < minPts:          # cannot happen (diagonal = eps); kept as a check
                raise AssertionError("phase-1 seed is not core")
            C += 1
            expand(seed, N, C)

    # Phase 2
    if budget_left():
        t0 = time.time()
        cnt2 = _phase2_counts(xy, eps, tree)
        maxrs_runs += 1
        if timings is not None:
            timings["phase2_squares_s"] = time.time() - t0
        for p in np.argsort(-cnt2, kind="stable"):
            if not budget_left() or cnt2[p] < minPts:
                break
            p = int(p)
            if labels[p] != _UNSEEN:
                continue
            outer += 1
            N = neighbors_of(p, eps)
            range_queries += 1
            if len(N) < minPts:
                labels[p] = -1           # may still join a cluster as border
                continue
            C += 1
            expand(p, N, C)

    out = np.array(labels, dtype=np.int64)
    out[out == _UNSEEN] = -1
    return out, outer, maxrs_runs, range_queries


def _tree_for(xy, tree):
    """The range-query tree to use: the one passed in, or a new R-tree."""
    if tree is None:
        tree = IndexiconIndex(xy)
    if tree.n != len(xy):
        raise ValueError(f"tree was built over {tree.n} points, DB has {len(xy)}")
    return tree

def DBSCAN_indexed(DB, eps, minPts, max_iterations=None, tree=None, progress=None):
    """Baseline: ordinary DBSCAN with every range query through a tree.

    Parameters
    ----------
    max_iterations : stop after this many clusters. None = all.
    tree : IndexiconIndex over DB. Pass the same one to DBSCAN_grid.
    progress : optional callable(points_visited, n, clusters, range_queries),
        called about every 1% of points, for long runs.

    Returns (labels, main_loop_iterations, range_queries).
    """
    xy = np.asarray(DB, dtype=float)
    n = len(xy)
    neighbors_of = _tree_for(xy, tree).neighbors_of
    report_every = max(n // 100, 1)
    labels = [_UNSEEN] * n          # a list: much faster than numpy per element
    C = 0
    iterations = 0
    range_queries = 0
 
    for p in range(n):
        iterations += 1
        if progress is not None and p % report_every == 0:
            progress(p, n, C, range_queries)
        if labels[p] != _UNSEEN:
            continue
        N = neighbors_of(p, eps)
        range_queries += 1
        if len(N) < minPts:
            labels[p] = -1
            continue
        if max_iterations is not None and C >= max_iterations:
            break
        C += 1
        labels[p] = C
        S = set(N)
        S.discard(p)
        while S:
            q = S.pop()
            if labels[q] == -1:        # was noise; reachable after all
                labels[q] = C
            if labels[q] != _UNSEEN:
                continue
            labels[q] = C
            Nq = neighbors_of(q, eps)
            range_queries += 1
            if len(Nq) >= minPts:
                S.update(Nq)

    if progress is not None:
        progress(n, n, C, range_queries)
    out = np.array(labels, dtype=np.int64)
    out[out == _UNSEEN] = -1
    return out, iterations, range_queries

def DBSCAN_grid(DB, eps, minPts, max_iterations=None, tree=None, grid=None, refine=1):
    """Our method: DBSCAN with densest-first seeds from a grid.

    Same clusters as ordinary DBSCAN; the grid only decides the order they
    are found in.

    Parameters
    ----------
    max_iterations : stop after this many clusters (top-k). None = all.
    tree : IndexiconIndex over DB, for every range query. Pass the same one
        to DBSCAN_indexed.
    grid : HierarchicalGridIndex whose eps range covers this eps, or a
        GridSeedIndex built for this eps. Either serves any minPts.
    refine : 1 = seed from the densest fine cell. m > 1 = take the m densest
        cells' candidates, range-query each, seed from the one with the most
        neighbours. Costs m queries per cluster; the winner's query is
        reused for its expansion.

    Returns (labels, outer_iterations, seed_calls, range_queries).
    """
    xy = np.asarray(DB, dtype=float)
    n = len(xy)
    neighbors_of = _tree_for(xy, tree).neighbors_of
    if grid is None:
        grid = GridSeedIndex(xy, eps)
    if grid.n != n:
        raise ValueError(f"grid was built over {grid.n} points, DB has {n}")
    fine_level, coarse_level = grid.levels_for(eps)    # raises if eps unsupported

    labels = [_UNSEEN] * n
    C = 0
    outer_iterations = 0
    seed_calls = 0
    range_queries = 0
 
    def query(p):
        nonlocal range_queries
        range_queries += 1
        return neighbors_of(p, eps)
 
    def expand(seed, N, cid):
        """Grow cluster cid from a seed already known to be core.
        N is the seed's neighbourhood, already queried by the caller."""
        labels[seed] = cid
        S = set(N)
        S.discard(seed)
        while S:
            q = S.pop()
            if labels[q] == -1:        # was noise; joins as a border point
                labels[q] = cid
            if labels[q] != _UNSEEN:
                continue
            labels[q] = cid
            Nq = query(q)
            if len(Nq) >= minPts:      # only core points keep it growing
                S.update(Nq)
 
    def budget_left():
        return max_iterations is None or C < max_iterations
 
    # Phase 1: seeds that are certainly core 
    fine = _Walk(fine_level, fine_level.by_count, fine_level.counts,
                 minPts, labels, _UNSEEN)
    while budget_left():
        seed_calls += 1
        if refine > 1:
            cands = fine_candidates(fine, refine)
            if not cands:
                break
            best, best_N = -1, None
            for c in cands:
                N = query(c)
                if best_N is None or len(N) > len(best_N):
                    best, best_N = c, N
            seed, N = best, best_N
        else:
            seed = fine_seed(fine)
            if seed < 0:
                break
            N = query(seed)
        outer_iterations += 1
        C += 1
        expand(seed, N, C)
 
    # Phase 2: candidates from 3x3 coarse blocks 
    coarse = _Walk(coarse_level, coarse_level.by_block, coarse_level.block,
                   minPts, labels, _UNSEEN)
    while budget_left():
        seed_calls += 1
        seed = coarse_seed(coarse)
        if seed < 0:
            break                       # no core points remain anywhere
        outer_iterations += 1
        N = query(seed)
        if len(N) < minPts:
            labels[seed] = -1           # not core; may still join as border later
            continue
        C += 1
        expand(seed, N, C)
 
    out = np.array(labels, dtype=np.int64)
    out[out == _UNSEEN] = -1
    return out, outer_iterations, seed_calls, range_queries