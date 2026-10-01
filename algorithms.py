"""
algorithms.py
=============
Incremental grid-based seed selection
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from spatial_index import RTreeIndex
from seed_grid import IncrementalGridSeeder

# DBSCAN
def RangeQuery(DB, distFunc, Q, eps):
    N = []
    Q_tuple = tuple(Q)
    for P in DB:
        P_tuple = tuple(P)
        if distFunc(Q_tuple, P_tuple) <= eps:
            N.append(P_tuple)
    return N

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def DBSCAN(DB, distFunc, eps, minPts, max_iterations=None):
    labels = {tuple(P): None for P in DB}
    C = 0
    main_loop_iterations = 0
    range_queries = 0
    clusters_found = 0

    for P in DB:
        main_loop_iterations += 1
        P_tuple = tuple(P)
        if labels[P_tuple] is not None:
            continue 

        N = RangeQuery(DB, distFunc, P, eps) 
        range_queries += 1

        if len(N) < minPts:
            labels[P_tuple] = -1
            continue

        if max_iterations is not None and clusters_found >= max_iterations:
            break 

        C += 1
        clusters_found += 1
        labels[P_tuple] = C  
        S = set(tuple(N)) - {P_tuple}

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)
    
    # Mark any remaining unlabeled points as noise
    for P in DB:
        P_tuple = tuple(P)
        if labels[P_tuple] is None:
            labels[P_tuple] = -1

    return labels, main_loop_iterations, range_queries

def DBSCAN_Optimized(DB, distFunc, eps, minPts, max_iterations=None):
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
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        x, y, best_sum, best_points = maxrs_sweepline_LP(
            unlabeled_list, rect_small, rect_small
        )
        maxrs_calls += 1

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

        S = set(N) - {P}
        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            unlabeled.discard(Q)
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    # ── Phase 2: 2*eps ───────────────────────────────────────────────────────
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        x, y, best_sum, best_points = maxrs_sweepline_LP(
            unlabeled_list, rect_large, rect_large
        )
        maxrs_calls += 1

        # No rectangle of this size is dense enough — all remaining are noise
        if best_sum < minPts:
            break

        phase2_iterations += 1

        P = tuple(best_points[0])

        N = RangeQuery(DB, distFunc, P, eps)
        range_queries += 1

        # Seed may be a lemon (not a core point) — mark noise and continue
        if len(N) < minPts:
            labels[P] = -1
            unlabeled.discard(P)
            continue

        C += 1
        labels[P] = C
        unlabeled.discard(P)

        S = set(N) - {P}
        while S:
            Q = S.pop()
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

def maxrs_sweepline_LP(points, rect_w, rect_h):
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
        return (0, 0, 0, [])
    
    # Check if points have weights
    has_weights = len(points[0]) == 3
    
    # Normalize points to always have weights
    if has_weights:
        normalized_points = points
    else:
        normalized_points = [(x, y, 1) for (x, y) in points]
    
    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(normalized_points)
    
    best_sum = -np.inf
    best_pos = None
    best_points = []

    # Add first element to list
    points_in_x_range = [points_by_x_val[0]]
    next_element_index = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    all_points_covered = False
    
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
        if next_element_index == len(points_by_x_val):
            all_points_covered = True

        # (2) Sort list by y-values
        # THEORY: I think this is very fast because the list is already mostly sorted
        points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

        # (3) Iterate until we reach a point in range vertically
        i = 0
        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i][1] < vertical_lb:
                i += 1

        # (4) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = current_pos
            # Collect points in best rectangle
            best_points = [
                pt for pt in points  # Use original points list
                if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
                and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
            ]
        
        # (5) Iterate through points, altering window as we go, until we use the current left-most point as bottom
        while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
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

        # (6) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i] != p:
                i += 1
            if i < len(points_in_x_range):
                del points_in_x_range[i]
        else:
            break
    
    return best_pos + (best_sum, best_points)

def DBSCAN_indexed(DB, distFunc, eps, minPts, max_iterations=None, rtree=None):
    """
    Vanilla DBSCAN with R-tree-backed range queries.

    `distFunc` is accepted for signature-compatibility with DBSCAN() but
    is unused here (the R-tree computes exact Euclidean distance itself).

    Pass a prebuilt `rtree` (an RTreeIndex) to reuse an index across
    repeated timing runs and exclude index-build time from the
    per-run measurement, matching how a real system would amortize
    index construction across many queries.
    """
    if rtree is None:
        rtree = RTreeIndex(DB)

    labels = {tuple(P): None for P in DB}
    C = 0
    main_loop_iterations = 0
    range_queries = 0
    clusters_found = 0

    for P in DB:
        main_loop_iterations += 1
        P_tuple = tuple(P)
        if labels[P_tuple] is not None:
            continue

        N = rtree.range_query(P_tuple, eps)
        range_queries += 1

        if len(N) < minPts:
            labels[P_tuple] = -1
            continue

        if max_iterations is not None and clusters_found >= max_iterations:
            break

        C += 1
        clusters_found += 1
        labels[P_tuple] = C
        S = set(N) - {P_tuple}

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            N = rtree.range_query(Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    for P in DB:
        P_tuple = tuple(P)
        if labels[P_tuple] is None:
            labels[P_tuple] = -1

    return labels, main_loop_iterations, range_queries

def DBSCAN_Optimized_indexed(DB, distFunc, eps, minPts, max_iterations=None,
                              rtree=None):
    """
    MaxRS-DBSCAN with R-tree-backed range queries.
    MaxRS unchanged from benchmark_topk.py.
    """
    if rtree is None:
        rtree = RTreeIndex(DB)

    labels = {tuple(P): None for P in DB}
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    phase1_iterations = 0
    phase2_iterations = 0
    maxrs_calls = 0
    range_queries = 0

    rect_small = eps / np.sqrt(2)
    rect_large = 2 * eps

    # ── Phase 1: eps/sqrt(2) ────────────────────────────────────────────
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        x, y, best_sum, best_points = maxrs_sweepline_LP(
            unlabeled_list, rect_small, rect_small
        )
        maxrs_calls += 1

        if best_sum < minPts:
            break

        phase1_iterations += 1
        P = tuple(best_points[0])

        N = rtree.range_query(P, eps)
        range_queries += 1

        if len(N) < minPts:
            labels[P] = -1
            unlabeled.discard(P)
            continue

        C += 1
        labels[P] = C
        unlabeled.discard(P)

        S = set(N) - {P}
        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            unlabeled.discard(Q)
            N = rtree.range_query(Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    # ── Phase 2: 2*eps ──────────────────────────────────────────────────
    while unlabeled and (max_iterations is None or C < max_iterations):
        unlabeled_list = list(unlabeled)
        x, y, best_sum, best_points = maxrs_sweepline_LP(
            unlabeled_list, rect_large, rect_large
        )
        maxrs_calls += 1

        if best_sum < minPts:
            break

        phase2_iterations += 1
        P = tuple(best_points[0])

        N = rtree.range_query(P, eps)
        range_queries += 1

        if len(N) < minPts:
            labels[P] = -1
            unlabeled.discard(P)
            continue

        C += 1
        labels[P] = C
        unlabeled.discard(P)

        S = set(N) - {P}
        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            unlabeled.discard(Q)
            N = rtree.range_query(Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    for P in unlabeled:
        labels[P] = -1

    outer_loop_iterations = phase1_iterations + phase2_iterations
    return labels, outer_loop_iterations, maxrs_calls, range_queries

def DBSCAN_grid(DB, distFunc, eps, minPts, max_iterations=None,
                rtree=None, seeder=None, n_shifts=4, refine=1):
    """
    Parameters
    ----------
    max_iterations : int or None
    refine : int 
        1 = seed from the densest grid cell directly. 
        >1 = have the grid propose that many candidates 
            and pick the true argmax by exact
            eps-count at `refine` extra range queries per cluster.
    """
    if rtree is None:
        rtree = RTreeIndex(DB)
    if seeder is None:
        seeder = IncrementalGridSeeder(DB, eps, minPts, n_shifts=n_shifts)
 
    labels = {tuple(P): None for P in DB}
    C = 0
    outer_iterations = 0
    seed_calls = 0
    range_queries = 0
 
    def expand(seed, cluster_id, skip_seed_query=False):
        """Standard DBSCAN expansion, R-tree backed."""
        nonlocal range_queries
        if skip_seed_query:
            N = rtree.range_query(seed, eps)
            range_queries += 1
        else:
            N = rtree.range_query(seed, eps)
            range_queries += 1
            if len(N) < minPts:
                return False
        labels[seed] = cluster_id
        seeder.remove(seed)
        S = set(N) - {seed}
        while S:
            Q = S.pop()
            if labels[Q] == -1:
                labels[Q] = cluster_id
            if labels[Q] is not None:
                continue
            labels[Q] = cluster_id
            seeder.remove(Q)
            Nq = rtree.range_query(Q, eps)
            range_queries += 1
            if len(Nq) >= minPts:
                S.update(Nq)
        return True
 
    while max_iterations is None or C < max_iterations:
        if refine > 1:
            cands = seeder.peek_core_candidates(refine)
            if not cands:
                seed = None
            else:
                best, best_n = None, -1
                for c in cands:
                    cnt = len(rtree.range_query(c, eps))
                    range_queries += 1
                    if cnt > best_n:
                        best, best_n = c, cnt
                seed = best
        else:
            seed = seeder.pop_guaranteed_core_seed()
        seed_calls += 1
        if seed is None:
            break
        outer_iterations += 1
        C += 1
        expand(seed, C, skip_seed_query=True)
 
    # Phase 2: candidate seeds from 3x3 coarse blocks 
    while max_iterations is None or C < max_iterations:
        seed = seeder.pop_candidate_seed()
        seed_calls += 1
        if seed is None:
            break                      # no core points remain
        outer_iterations += 1
        N = rtree.range_query(seed, eps)
        range_queries += 1
        if len(N) < minPts:
            labels[seed] = -1          # lemon seed
            seeder.remove(seed)
            continue
        C += 1
        labels[seed] = C
        seeder.remove(seed)
        S = set(N) - {seed}
        while S:
            Q = S.pop()
            if labels[Q] == -1:
                labels[Q] = C
            if labels[Q] is not None:
                continue
            labels[Q] = C
            seeder.remove(Q)
            Nq = rtree.range_query(Q, eps)
            range_queries += 1
            if len(Nq) >= minPts:
                S.update(Nq)
 
    for P in DB:
        pt = tuple(P)
        if labels[pt] is None:
            labels[pt] = -1
 
    return labels, outer_iterations, seed_calls, range_queries