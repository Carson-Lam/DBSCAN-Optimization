"""
seed_grid.py
=============
Incremental grid-based seed selection

The old version used MaxRS and took a full sweep over all remaining points, once
per cluster. This does the same job with a grid built once up front, so
picking a seed is just a heap pop.
 
Two grids:
 
fine:  
        cells of side eps/sqrt(2). The diagonal works out to exactly
        eps, so all points in one cell are within eps of each other.
        That means a cell with >= minPts points contains only core
        points, and we can seed from it without checking.
 
coarse: 
        cells of side eps scored by the 3x3 block around them. Anything
        within eps of a point in cell c is somewhere in c's 3x3 block,
        so the block total is an upper bound on any neighbourhood in c.
        When no block in the 3x3 reaches minPts, there are no core points left and
        we can stop.
 
fine_all / block_all:           
        how dense is this region (fixed)

fine_live / coarse_live_count:
        how many points are left in each grid
 
Because the heap keys come from the fixed counts, we build the heaps once
and never push to them again.
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

    # Optional extra copies of the fine grid, shifted by half a cell.
    # A cluster sitting on a cell corner gets split across 4 cells but
    # a shifted copy of the grid will catch it centred. This means
    # it only affects which seed we pick first and not the final clustering.
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
        self.fine_side = eps / math.sqrt(2.0)    # diagonal == eps
        s = self.fine_side
        self.offsets = [(dx * s, dy * s) for dx, dy in self._OFFSET_SETS[n_shifts]]
        G = len(self.offsets)

        # One fine grid per shift.
        self.fine_of = [dict() for _ in range(G)]              # point -> cell
        self.fine_all = [defaultdict(int) for _ in range(G)]   # cell -> count
        self.fine_live = [defaultdict(set) for _ in range(G)]  # cell -> unlabeled
        self._members = set()

        # One coarse grid.
        self.coarse_of = {}
        self.coarse_live_count = defaultdict(int)
        self.block_all = defaultdict(int)

        for p in points:
            pt = tuple(p)
            if pt in self._members:                # skip duplicate coordinates     
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

            # Bump all 9 cells around this one, so block_all[d] ends up as
            # the number of points in the 3x3 block centred on d.
            for d in self._neighbourhood(cc):
                self.block_all[d] += 1

        self.n_live = len(self._members)

        # heapq is a min-heap, so counts go in negated.
        # Cells under minPts are left out because they can never seed anything.
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
        """Which cell of grid g holds p."""
        s = self.fine_side
        ox, oy = self.offsets[g]
        return (int(math.floor((p[0] - ox) / s)),
                int(math.floor((p[1] - oy) / s)))

    def _coarse_cell(self, p):
        """Which coarse cell holds p."""
        s = self.eps
        return (int(math.floor(p[0] / s)), int(math.floor(p[1] / s)))

    @staticmethod
    def _neighbourhood(c):
        """The 9 cells of the 3x3 block around c, c included."""
        i, j = c
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                yield (i + di, j + dj)

    # ------------------------------------------------------------------
    def remove(self, p):
        """Called once per point as DBSCAN labels it, so we stop offering
        it as a seed. Doesn't touch the counts or the heaps. """
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
        """A still-unlabeled point from the densest fine cell.
 
        It's definitely a core point because we know the fine cell diagonal
        is eps and the cell has at least minPts points.
 
        None means no dense fine cell has unlabeled points left. That ends
        phase 1, but there may still be clusters around 
        """
        h = self.fine_heap
        while h:
            neg, g, cell = h[0]
            if -neg < self.minPts:
                return None               # biggest cell is too small, so all are
            live = self.fine_live[g].get(cell)
            if not live:
                heapq.heappop(h)          # all used up by an earlier cluster
                continue
            return next(iter(live))       # any of them will do
        return None

    def peek_core_candidates(self, m):
        """The top m cells' worth of candidate seeds, without using them up.
 
        A cell count is only a rough density estimate because the cell is
        smaller than an eps-neighbourhood, so where the cluster sits
        relative to the grid matters. The caller range-queries each of
        these cells to find the densest one.
        """
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
                    heapq.heappop(h)              # used up, drop it
                    continue
                scratch.append(heapq.heappop(h))  # park it, restore below
                pt = next(iter(live))
                if pt not in seen:                # shifted grids can repeat
                    seen.add(pt)
                    out.append(pt)
        finally:
            for e in scratch:
                heapq.heappush(h, e)
        return out

    def pop_candidate_seed(self):
        """A still-unlabeled point from the best 3x3 coarse block.
 
        Picks up clusters phase 1 missed because they were spread across
        cell boundaries. The point is only a candidate because the 
        3x3 block point total is an upper bound so the caller still needs
        to range-query it to see if it's actually a core point.

        None means no core points are left anywhere, so everything
        remaining is noise.
        """
        h = self.block_heap
        while h:
            neg, cell = h[0]
            if -neg < self.minPts:
                return None               
            pt = self._any_live_in_block(cell)
            if pt is None:
                heapq.heappop(h)          # nothing unlabeled in this block
                continue
            return pt
        return None

    def _any_live_in_block(self, coarse_cell):
        """Any unlabeled point in the 3x3 block around coarse_cell.
        It doesn't have to be in coarse_cell itself."""
        for d in self._neighbourhood(coarse_cell):
            if self.coarse_live_count.get(d, 0) > 0:
                pt = self._any_live_in_coarse(d)
                if pt is not None:
                    return pt
        return None

    def _any_live_in_coarse(self, coarse_cell):
        """Any unlabeled point in one specific coarse cell.
 
        We don't keep a cell to points index for the coarse grid, so we check
        the fine cells that overlap it instead. Grid 0 is the unshifted
        one, so every point is in exactly one of its cells.
        """
        ratio = self.eps / self.fine_side                # sqrt(2) fine cells per coarse
        i, j = coarse_cell
        lo_i = int(math.floor(i * ratio)) - 1            # +/-1 for boundary rounding
        hi_i = int(math.ceil((i + 1) * ratio)) + 1
        lo_j = int(math.floor(j * ratio)) - 1
        hi_j = int(math.ceil((j + 1) * ratio)) + 1
        live0 = self.fine_live[0]                        # grid 0 covers all points
        for fi in range(lo_i, hi_i + 1):
            for fj in range(lo_j, hi_j + 1):
                live = live0.get((fi, fj))
                if live:
                    for pt in live:
                        # a fine cell can overlap two coarse ones, so check
                        if self.coarse_of.get(pt) == coarse_cell:
                            return pt
        return None