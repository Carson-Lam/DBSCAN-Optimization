"""
dbscan_indexed.py
==================
- R-tree-indexed variants of DBSCAN and MaxRS-DBSCAN (GDBSCAN).
- Imports from benchmark_topk.py
- Indexes the `RangeQuery` calls where both vanilla DBSCAN and 
MaxRS-DBSCAN spend their per-point O(n) cost.

Dependencies: numpy, rtree (pip install rtree)
"""

import numpy as np

from benchmark_topk import maxrs_sweepline_LP
from spatial_index import RTreeIndex


# ═══════════════════════════════════════════════════════════════════════════
# Indexed vanilla DBSCAN
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Indexed MaxRS-DBSCAN (GDBSCAN)
# ═══════════════════════════════════════════════════════════════════════════

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