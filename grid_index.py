"""
grid_index.py
==============
A vectorised index-based version of seed_grid.py.

Same idea as seed_grid.py. A fine grid that certifies core points and a
coarse grid that proves when no core points are left but with four changes:

1. Built with numpy instead of a Python loop over every point so it can
   be built on tens of millions of points.

2. Points are identified by their position in the input, not by their
   coordinates. Several points at the same coordinates (e.g. businesses
   in one building) now count separately. seed_grid.py deduplicated them,
   which made its counts disagree with the range queries and could make
   the "no core points left" test fire while core points remained.

3. Cells are kept in arrays presorted by count, instead of heaps. The
   counts never change, so the order never changes either. 

4. No shifted copies of the fine grid (n_shifts). We have to stack
   these grids into a hierarchy and shifted grids don't nest. Use
   `refine` in DBSCAN_grid_ids to improve seed order instead.

What is built once (GridSeedIndex, depends only on the points and eps):
    per grid: each cell's point count, the points in each cell, and the
    cells sorted by count; for the coarse grid also each cell's 3x3 total.

What is per query (inside DBSCAN_grid_ids): 
    which points are labelled, and how far down each sorted array we have walked.
"""

import numpy as np

SQRT2 = np.sqrt(2.0)

# Cell sides get nudged by this factor in the direction that keeps each
# guarantee safe under floating-point rounding: the fine side a hair
# smaller (diagonal stays <= eps), the coarse side a hair larger (one ring
# of cells still covers eps).
_NUDGE = 1e-9


class GridLevel:
    """One grid of square cells over a point set. Built once, never changed."""

    def __init__(self, xy, side, origin):
        xy = np.asarray(xy, dtype=float)
        self.side = float(side)
        self.origin = np.asarray(origin, dtype=float)

        # integer cell coordinates, shifted by +1 so a cell's neighbours
        # (i-1, j-1) are never negative
        ij = np.floor((xy - self.origin) / self.side).astype(np.int64) + 1
        # one integer key per cell: key = i * ncols + j. ncols leaves room
        # for j+1, so a neighbour key never wraps into the next row.
        self.ncols = int(ij[:, 1].max()) + 2
        keys = ij[:, 0] * self.ncols + ij[:, 1]

        # sort points by cell; the points of each cell end up contiguous
        self.point_order = np.argsort(keys, kind="stable")
        self.cell_keys, self.starts, self.counts = np.unique(
            keys[self.point_order], return_index=True, return_counts=True)
        self.n_cells = len(self.cell_keys)

        # cells sorted densest first; equal counts ordered by cell key so
        # runs are reproducible
        self.by_count = np.lexsort((self.cell_keys, -self.counts))

        # 3x3 block total for every cell: add up the counts of the 9 cells
        # around it. Only the coarse grid uses this, but it's cheap.
        self.block = np.zeros(self.n_cells, dtype=np.int64)
        for d in self._offsets():
            pos, hit = self._find(self.cell_keys + d)
            self.block[hit] += self.counts[pos[hit]]
        self.by_block = np.lexsort((self.cell_keys, -self.block))

    def _offsets(self):
        """Key differences to the 9 cells of a 3x3 block (centre included)."""
        nc = self.ncols
        return [di * nc + dj for di in (-1, 0, 1) for dj in (-1, 0, 1)]

    def _find(self, keys):
        """Cell index for each key, and whether that cell exists."""
        pos = np.searchsorted(self.cell_keys, keys)
        pos = np.minimum(pos, self.n_cells - 1)
        return pos, self.cell_keys[pos] == keys

    def neighbours(self, c):
        """Indices of the existing cells in the 3x3 block around cell c."""
        k = self.cell_keys[c]
        out = []
        for d in self._offsets():
            nk = k + d
            pos = int(np.searchsorted(self.cell_keys, nk))
            if pos < self.n_cells and self.cell_keys[pos] == nk:
                out.append(pos)
        return out


class GridSeedIndex:
    """The two grids DBSCAN_grid_ids needs for one eps.

    Reusable across any number of queries and any minPts at that eps.
    """

    def __init__(self, xy, eps):
        xy = np.asarray(xy, dtype=float)
        self.eps = float(eps)
        self.n = len(xy)
        origin = xy.min(axis=0)
        # fine: diagonal <= eps, so a cell with >= minPts points is all core
        self.fine = GridLevel(xy, eps / SQRT2 * (1 - _NUDGE), origin)
        # coarse: side >= eps, so a 3x3 block covers every point's eps-reach
        self.coarse = GridLevel(xy, eps * (1 + _NUDGE), origin)


class _Walk:
    """Per-query position in one grid's sorted cell array.

    `pos` only moves forward: once every point in a cell is labelled, the
    cell can never offer a seed again. Each cell also keeps a cursor to its
    first possibly-unlabelled point, which also only moves forward, so
    scanning for live points costs O(n) in total over a whole query.
    """

    def __init__(self, level, order, score, minPts, labels, unseen):
        self.level = level
        self.order = order
        self.score = score
        self.minPts = minPts
        self.labels = labels
        self.unseen = unseen
        self.pos = 0
        self.cursor = level.starts.copy()

    def live_point(self, c):
        """First unlabelled point in cell c, or -1."""
        lv = self.level
        end = lv.starts[c] + lv.counts[c]
        cur = self.cursor[c]
        po, labels, unseen = lv.point_order, self.labels, self.unseen
        while cur < end and labels[po[cur]] != unseen:
            cur += 1
        self.cursor[c] = cur
        return int(po[cur]) if cur < end else -1

    def at_end(self):
        return (self.pos >= len(self.order)
                or self.score[self.order[self.pos]] < self.minPts)


def fine_seed(walk):
    """Live point from the densest fine cell with >= minPts points, or -1.

    That point is certainly core. -1 ends phase 1 but does not mean the
    clusters are all found: phase 2 catches clusters split across cells.
    """
    while not walk.at_end():
        p = walk.live_point(walk.order[walk.pos])
        if p >= 0:
            return p
        walk.pos += 1          # everything in this cell is labelled
    return -1


def fine_candidates(walk, m):
    """Up to m live points, one each from the m densest fine cells that
    still have one. Does not move past cells that still have live points."""
    out = []
    q = walk.pos
    order, score, minPts = walk.order, walk.score, walk.minPts
    while q < len(order) and len(out) < m and score[order[q]] >= minPts:
        p = walk.live_point(order[q])
        if p >= 0:
            out.append(p)
        elif q == walk.pos:
            walk.pos += 1      # exhausted cell at the front: drop it
        q += 1
    return out


def coarse_seed(walk):
    """Live point from the coarse cell with the largest 3x3 total, or -1.

    Only a candidate: the caller must range-query it. -1 means no
    unlabelled point anywhere can be core, so the rest is noise.
    """
    lv = walk.level
    while not walk.at_end():
        c = walk.order[walk.pos]
        for nb in lv.neighbours(c):
            p = walk.live_point(nb)
            if p >= 0:
                return p
        walk.pos += 1          # nothing unlabelled in this whole block
    return -1