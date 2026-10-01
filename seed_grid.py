"""
seed_grid.py
=============
Incremental grid-based seed selection
"""

import heapq
import math
from collections import defaultdict


class IncrementalGridSeeder:
    """
        Several grids over a point set. 
        Live point sets shrink.
        Seed candidates ordered by density.
    """

    __slots__ = ("eps", "minPts", "fine_side", "offsets",
                 "fine_of", "fine_all", "fine_live", "fine_heap",
                 "coarse_of", "coarse_live_count", "block_all",
                 "block_heap", "n_live", "_members")

    _OFFSET_SETS = {
        1: [(0.0, 0.0)],
        2: [(0.0, 0.0), (0.5, 0.5)],
        4: [(0.0, 0.0), (0.5, 0.0), (0.0, 0.5), (0.5, 0.5)],
    }

    def __init__(self, points, eps, minPts, n_shifts=4):
        if n_shifts not in self._OFFSET_SETS:
            raise ValueError(f"n_shifts must be one of {sorted(self._OFFSET_SETS)}")
        self.eps = eps
        self.minPts = minPts
        self.fine_side = eps / math.sqrt(2.0)
        s = self.fine_side
        self.offsets = [(dx * s, dy * s) for dx, dy in self._OFFSET_SETS[n_shifts]]
        G = len(self.offsets)

        self.fine_of = [dict() for _ in range(G)]        # g: point -> cell
        self.fine_all = [defaultdict(int) for _ in range(G)]
        self.fine_live = [defaultdict(set) for _ in range(G)]
        self._members = set()                            # dedup of points

        self.coarse_of = {}
        self.coarse_live_count = defaultdict(int)
        self.block_all = defaultdict(int)

        for p in points:
            pt = tuple(p)
            if pt in self._members:                      
                continue
            self._members.add(pt)
            for g in range(G):
                fc = self._fine_cell(pt, g)
                self.fine_of[g][pt] = fc
                self.fine_all[g][fc] += 1
                self.fine_live[g][fc].add(pt)

            cc = self._coarse_cell(pt)
            self.coarse_of[pt] = cc
            self.coarse_live_count[cc] += 1
            for d in self._neighbourhood(cc):
                self.block_all[d] += 1

        self.n_live = len(self._members)

        self.fine_heap = [(-c, g, k)
                          for g in range(G)
                          for k, c in self.fine_all[g].items()
                          if c >= minPts]
        heapq.heapify(self.fine_heap)
        self.block_heap = [(-v, k) for k, v in self.block_all.items()
                           if v >= minPts]
        heapq.heapify(self.block_heap)

    # ------------------------------------------------------------------
    def _fine_cell(self, p, g):
        s = self.fine_side
        ox, oy = self.offsets[g]
        return (int(math.floor((p[0] - ox) / s)),
                int(math.floor((p[1] - oy) / s)))

    def _coarse_cell(self, p):
        s = self.eps
        return (int(math.floor(p[0] / s)), int(math.floor(p[1] / s)))

    @staticmethod
    def _neighbourhood(c):
        i, j = c
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                yield (i + di, j + dj)

    # ------------------------------------------------------------------
    def remove(self, p):
        pt = tuple(p)
        if pt not in self._members:
            return
        self._members.discard(pt)
        for g in range(len(self.offsets)):
            fc = self.fine_of[g].pop(pt, None)
            if fc is not None:
                self.fine_live[g][fc].discard(pt)
        cc = self.coarse_of.pop(pt)
        self.coarse_live_count[cc] -= 1
        self.n_live -= 1

    def remove_many(self, pts):
        for p in pts:
            self.remove(p)

    # ------------------------------------------------------------------
    def pop_guaranteed_core_seed(self):
        h = self.fine_heap
        while h:
            neg, g, cell = h[0]
            if -neg < self.minPts:
                return None
            live = self.fine_live[g].get(cell)
            if not live:
                heapq.heappop(h)          # cell exhausted
                continue
            return next(iter(live))
        return None

    def peek_core_candidates(self, m):
        h = self.fine_heap
        out = []
        seen = set()
        scratch = []
        try:
            while h and len(out) < m:
                neg, g, cell = h[0]
                if -neg < self.minPts:
                    break
                live = self.fine_live[g].get(cell)
                if not live:
                    heapq.heappop(h)              # genuinely exhausted
                    continue
                scratch.append(heapq.heappop(h))  # set aside, restore later
                pt = next(iter(live))
                if pt not in seen:
                    seen.add(pt)
                    out.append(pt)
        finally:
            for e in scratch:
                heapq.heappush(h, e)
        return out

    def pop_candidate_seed(self):
        h = self.block_heap
        while h:
            neg, cell = h[0]
            if -neg < self.minPts:
                return None               # sound termination
            pt = self._any_live_in_block(cell)
            if pt is None:
                heapq.heappop(h)          # whole block exhausted
                continue
            return pt
        return None

    def _any_live_in_block(self, coarse_cell):
        for d in self._neighbourhood(coarse_cell):
            if self.coarse_live_count.get(d, 0) > 0:
                pt = self._any_live_in_coarse(d)
                if pt is not None:
                    return pt
        return None

    def _any_live_in_coarse(self, coarse_cell):
        ratio = self.eps / self.fine_side                # sqrt(2)
        i, j = coarse_cell
        lo_i = int(math.floor(i * ratio)) - 1
        hi_i = int(math.ceil((i + 1) * ratio)) + 1
        lo_j = int(math.floor(j * ratio)) - 1
        hi_j = int(math.ceil((j + 1) * ratio)) + 1
        live0 = self.fine_live[0]                        # grid 0 covers all points
        for fi in range(lo_i, hi_i + 1):
            for fj in range(lo_j, hi_j + 1):
                live = live0.get((fi, fj))
                if live:
                    for pt in live:
                        if self.coarse_of.get(pt) == coarse_cell:
                            return pt
        return None