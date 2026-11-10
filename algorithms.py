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

  DBSCAN_single_maxrs  Single-scan MaxRS version (Lance; in progress).

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

import numpy as np

from spatial_index import IndexiconIndex
from grid_index import GridSeedIndex, _Walk, fine_seed, fine_candidates, coarse_seed

_UNSEEN = -2   # not visited yet; turned into -1 (noise) before returning


def single_run_maxrs(points, rect_w, rect_h):
    """
    Sweep line O(n² log n) version.
    
    Key optimization: For each x-position, filter points ONCE,
    then sweep through y-positions checking only filtered points.
    
    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
    Returns:
        (best_x, best_y, max_sum)
    """
    if not points:
        #return (0, 0, 0, [])
        return {}
    
    # Check if points have weights
    has_weights = len(points[0]) == 3
    
    # Normalize points to always have weights
    if has_weights:
        normalized_points = points
    else:
        normalized_points = [(x, y, 1) for (x, y) in points]
    
    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(normalized_points)
    
    # best_sum = -np.inf
    # best_pos = None
    # best_points = []

    # Map points -> [sum of points in RS, position of RS, list of points in RS]
    results_per_point = {}

    # Add first element to list
    points_in_x_range = [points_by_x_val[0]]
    next_element_index = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    # I don't think this holds anymore. A point might be completely excluded if we do this
    #all_points_covered = False
    
    # Start left side of grid at x's
    # Output: X jumps [1,5, 6, 8, ...]
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h

        # (1) Add new points to list
        while next_element_index < len(points_by_x_val) and points_by_x_val[next_element_index][0] <= horizontal_ub:
            points_in_x_range.append(points_by_x_val[next_element_index])
            next_element_index += 1

        # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
        # if next_element_index == len(points_by_x_val):
        #     all_points_covered = True

        # (2) Sort list by y-values
        # THEORY: I think this is very fast because the list is already mostly sorted
        points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

        # (3) Iterate until we reach a point in range vertically
        i = 0
        #if not all_points_covered:
        while i < len(points_in_x_range) and points_in_x_range[i][1] < vertical_lb:
            i += 1

        # (4) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
        # Moved this from outer loop, now should track best for each point in innter loop
        best_sum = -np.inf
        best_pos = None
        best_points = []

        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        # No need for commented section now, can directly set as "best" for the left anchor
        best_sum = current_sum
        best_pos = current_pos
        best_points = [
            pt for pt in points
            if best_pos[0] <= pt[0] <= best_pos[0] + rect_w
            and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
        ]
        # if current_sum > best_sum:
        #     best_sum = current_sum
        #     best_pos = current_pos
        #     # Collect points in best rectangle
        #     best_points = [
        #         pt for pt in points  # Use original points list
        #         if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
        #         and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
        #     ]
        
        # (5) Iterate through points, altering window as we go, until we use the current left-most point as bottom
        # while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
        while j < len(points_in_x_range) and points_in_x_range[i] != p:
            # Remove weight of previous bottom point
            current_sum -= points_in_x_range[i][2]
            # Shift i to next point and reset current pos
            i += 1
            current_pos = (p[0], points_in_x_range[i][1])
            # Shift j and add new points
            while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
                current_sum += points_in_x_range[j][2]
                j += 1
            # Check new window
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = current_pos
                # Collect points in best rectangle
                best_points = [
                    pt for pt in points  # Use original points list
                    if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
                    and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
                ]

        # (6) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
        #if not all_points_covered:
        while i < len(points_in_x_range) and points_in_x_range[i] != p:
            i += 1
        if i < len(points_in_x_range):
            del points_in_x_range[i]

        # (7) Finally, mark the result for the RS anchored on the left by this point        
        results_per_point[p] = best_pos + (best_sum, best_points)

    # Sort results by densest RS
    results_per_point = dict(sorted(results_per_point.items(), 
                                    key=lambda item: item[1][2], # item[1] is the dict value, item[1][2] accesses the sum
                                    reverse=True))
    
    return results_per_point

# TODO(Lance): doesn't run yet. It calls RangeQuery, which is no longer in
# this file, and uses unlabeled_list before assigning it. To compare
# fairly with the other two it should also identify points by position
# and take the same `tree` for its range queries.
def DBSCAN_single_maxrs(DB, distFunc, eps, minPts, max_iterations=None):
    """
    Optimized DBSCAN using MaxRS to find densest regions first.
    
    Two-phase approach:
    Phase 1: MaxRS with rect side eps/sqrt(2). Any region found GUARANTEES
             all points inside are core points (max pairwise distance = eps).
             No "lemon" seeds possible.
    Phase 2: MaxRS with rect side 2*eps. Finds candidate regions that may
             contain core points, but seed point may still be noise.
             Terminates when no 2*eps rectangle has >= minPts points,
             guaranteeing all remaining unlabeled points are noise.
    """
    labels = {tuple(P): None for P in DB}
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    # Counters
    phase1_iterations = 0
    phase2_iterations = 0
    maxrs_calls = 0
    range_queries = 0

    rect_small = eps / np.sqrt(2)   # Phase 1: guaranteed core points
    rect_large = 2 * eps            # Phase 2: conservative superset

    # ── Phase 1: eps/sqrt(2) ─────────────────────────────────────────────────
    results_per_point = single_run_maxrs(unlabeled_list, rect_small, rect_small)
    maxrs_calls += 1
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        
        # x, y, best_sum, best_points = maxrs_sweepline_LP(
        #     unlabeled_list, rect_small, rect_small
        # )

        # Get densest RS
        densest_point, val = next(iter(results_per_point.items()))
        x, y, best_sum, best_points = val
        # Remove from dict
        results_per_point.pop(densest_point, None)

        # No rectangle of this size is dense enough — exit Phase 1
        if best_sum < minPts:
            break

        phase1_iterations += 1

        # Every point in this rectangle is guaranteed to be a core point,
        # so we can pick any of them as seed — pick the first
        P = tuple(best_points[0])

        N = RangeQuery(DB, distFunc, P, eps)
        range_queries += 1

        # Should always pass in Phase 1, but keep as sanity check
        if len(N) < minPts:
            labels[P] = -1
            unlabeled.discard(P)
            continue

        C += 1
        labels[P] = C
        unlabeled.discard(P)
        results_per_point.pop(P, None)

        S = set(N) - {P}
        while S:
            Q = S.pop()
            results_per_point.pop(Q, None)
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            unlabeled.discard(Q)
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    # ── Phase 2: 2*eps ───────────────────────────────────────────────────────
    results_per_point = single_run_maxrs(unlabeled_list, rect_large, rect_large)
    maxrs_calls += 1
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        
        # x, y, best_sum, best_points = maxrs_sweepline_LP(
        #     unlabeled_list, rect_large, rect_large
        # )

        # Get densest RS
        densest_point, val = next(iter(results_per_point.items()))
        x, y, best_sum, best_points = val
        # Remove from dict
        results_per_point.pop(densest_point, None)

        # No rectangle of this size is dense enough — all remaining are noise
        if best_sum < minPts:
            break

        phase2_iterations += 1

        # Cycle points in rectangle
        for pt in best_points:
            P = tuple(pt)
            # Ignore if we've expanded already
            if labels[P] is not None: continue

            # Run range query and discard if not a core
            N = RangeQuery(DB, distFunc, P, eps)
            range_queries += 1
            if len(N) < minPts:
                labels[P] = -1
                unlabeled.discard(P)
                results_per_point.pop(P, None)
                continue

            # Otherwise, expand core point
            C += 1
            labels[P] = C
            unlabeled.discard(P)
            results_per_point.pop(P, None)

            S = set(N) - {P}
            while S:
                Q = S.pop()
                results_per_point.pop(Q, None)
                if labels[Q] == -1: labels[Q] = C
                if labels[Q] is not None: continue
                labels[Q] = C
                unlabeled.discard(Q)
                N = RangeQuery(DB, distFunc, Q, eps)
                range_queries += 1
                if len(N) >= minPts:
                    S.update(N)

    # Mark all remaining unlabeled points as noise
    for P in unlabeled:
        labels[P] = -1

    outer_loop_iterations = phase1_iterations + phase2_iterations
    return labels, outer_loop_iterations, maxrs_calls, range_queries

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