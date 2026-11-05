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

This file has two indexes:
  GridSeedIndex(xy, eps)
      Two grids built for one eps: fine side eps/sqrt(2), coarse side eps.
 
  HierarchicalGridIndex(xy, eps_min, eps_max)
      A ladder of grids with sides base, 2*base, 4*base, ... built once.
      Serves any minPts and any eps in [eps_min, eps_max]. For a query eps
      it picks two rungs:
          fine   = largest rung with side <= eps/sqrt(2)
          coarse = smallest rung with side >= eps
      A smaller fine cell still has diagonal <= eps, and a larger coarse
      cell still clears eps, so both guarantees hold at every eps in range.

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
        # (cell_keys is already ascending, so a stable sort on -count gives
        # the same order as sorting on (-count, key) and is much faster)
        self.by_count = np.argsort(-self.counts, kind="stable")
 
        # 3x3 block total for every cell: add up the counts of the 9 cells
        # around it. Only the coarse grid uses this, but it's cheap.
        self.block = np.zeros(self.n_cells, dtype=np.int64)
        for d in self._offsets():
            pos, hit = self._find(self.cell_keys + d)
            self.block[hit] += self.counts[pos[hit]]
        self.by_block = np.argsort(-self.block, kind="stable")
 
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
 
    def levels_for(self, eps):
        """(fine, coarse) grids for this eps. Only the eps it was built for."""
        if eps != self.eps:
            raise ValueError(
                f"GridSeedIndex was built for eps={self.eps}, query uses "
                f"eps={eps}. Use HierarchicalGridIndex to serve several eps.")
        return self.fine, self.coarse
 
    def nbytes(self):
        """Approximate memory held by the index, in bytes."""
        return sum(_level_nbytes(lv) for lv in (self.fine, self.coarse))
 
 
# ════════════════════════════════════════════════════════════════════════════
# Hierarchical grid: one index for a whole range of eps
# ════════════════════════════════════════════════════════════════════════════
 
# Selection margin: a rung must clear its guarantee by this relative margin.
# Without it, an eps landing exactly on a rung side could, through
# floating-point rounding, put two points at distance exactly eps two cells
# apart. With it, such an eps uses the next rung up for the coarse grid,
# which is safe and costs almost nothing.

_TOL = 1e-9
 
_M1 = np.uint64(0x5555555555555555)
_M2 = np.uint64(0x3333333333333333)
_M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
_M8 = np.uint64(0x00FF00FF00FF00FF)
_M16 = np.uint64(0x0000FFFF0000FFFF)
_M32 = np.uint64(0x00000000FFFFFFFF)
_U = [np.uint64(k) for k in (0, 1, 2, 4, 8, 16)]
 
# finest-grid cell coordinates must stay below this (Morton keys are 64-bit)
_MAX_CELLS_PER_AXIS = 2 ** 31
 
 
def _spread(v):
    """Put a 0 bit between each bit of v (uint64 array, values < 2^32)."""
    v = v & _M32
    v = (v | (v << _U[5])) & _M16
    v = (v | (v << _U[4])) & _M8
    v = (v | (v << _U[3])) & _M4
    v = (v | (v << _U[2])) & _M2
    v = (v | (v << _U[1])) & _M1
    return v
 
 
def _squeeze(v):
    """Inverse of _spread: take every other bit."""
    v = v & _M1
    v = (v | (v >> _U[1])) & _M2
    v = (v | (v >> _U[2])) & _M4
    v = (v | (v >> _U[3])) & _M8
    v = (v | (v >> _U[4])) & _M16
    v = (v | (v >> _U[5])) & _M32
    return v
 
 
def morton(i, j):
    """Z-order key of integer cell coordinates (i, j). Arrays in, uint64 out.
 
    Interleaving the bits of i and j gives every cell one 64-bit key with
    a useful property: dropping the last 2 bits gives the key of the cell
    one rung up (twice the side) that contains it. So if points are sorted
    by their finest-rung key, the points of ANY cell on ANY rung are a
    contiguous run of that one sorted array.
    """
    i = np.asarray(i, dtype=np.uint64)
    j = np.asarray(j, dtype=np.uint64)
    return _spread(i) | (_spread(j) << _U[1])
 
 
def unmorton(k):
    """(i, j) integer coordinates of Morton keys k."""
    k = np.asarray(k, dtype=np.uint64)
    return (_squeeze(k).astype(np.int64),
            _squeeze(k >> _U[1]).astype(np.int64))
 
 
_OFFSETS_3x3 = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
 
 
class HierLevel:
    """One rung of a HierarchicalGridIndex.
 
    Looks like a GridLevel to the seed walkers (same attributes, same
    neighbours()), but its cells are Morton keys and its points live in the
    point array shared by every rung.
    """
 
    def __init__(self, rung, side, point_order, keys, starts, counts,
                 as_fine=True, as_coarse=True):
        self.rung = rung
        self.side = float(side)
        self.point_order = point_order       # shared by all rungs
        self.cell_keys = keys                # sorted uint64
        self.starts = starts                 # into point_order
        self.counts = counts
        self.n_cells = len(keys)
        self.as_fine = as_fine
        self.as_coarse = as_coarse
 
        # Only build what this rung can be used for. Rung 0 is too small to
        # ever be a coarse grid and the top rungs too big to ever be fine,
        # and the small rungs hold about one cell per point, so skipping
        # their unused arrays roughly halves the build.
        self.by_count = self.block = self.by_block = None
 
        # densest first; ties broken by cell key (keys are ascending, so a
        # stable sort keeps them in key order) so runs are reproducible
        if as_fine:
            self.by_count = np.argsort(-counts, kind="stable").astype(starts.dtype)
        if not as_coarse:
            return
 
        # 3x3 block totals
        ci, cj = unmorton(keys)
        block = np.zeros(self.n_cells, dtype=counts.dtype)
        for di, dj in _OFFSETS_3x3:
            ni, nj = ci + di, cj + dj
            ok = np.flatnonzero((ni >= 0) & (nj >= 0))
            nk = morton(ni[ok], nj[ok])
            pos = np.minimum(np.searchsorted(keys, nk), self.n_cells - 1)
            hit = keys[pos] == nk
            block[ok[hit]] += counts[pos[hit]]
        self.block = block
        self.by_block = np.argsort(-block, kind="stable").astype(starts.dtype)
 
    def neighbours(self, c):
        """Indices of the existing cells in the 3x3 block around cell c."""
        ci, cj = unmorton(self.cell_keys[c:c + 1])
        ni = ci[0] + np.array([d[0] for d in _OFFSETS_3x3])
        nj = cj[0] + np.array([d[1] for d in _OFFSETS_3x3])
        ok = (ni >= 0) & (nj >= 0)
        nk = morton(ni[ok], nj[ok])
        pos = np.minimum(np.searchsorted(self.cell_keys, nk), self.n_cells - 1)
        return pos[self.cell_keys[pos] == nk].tolist()
 
 
class HierarchicalGridIndex:
    """Grids at sides base * 2^r, r = 0..R, serving every eps in
    [eps_min, eps_max] and every minPts.
 
    The rungs are chosen so that:
      - eps_min gets rung 0 as its fine grid (base = eps_min / sqrt(2)),
      - eps_max has a coarse grid (top rung side >= eps_max).
    So the number of rungs is about log2(eps_max / eps_min) + 2.
 
    Memory: one shared point order (n integers) plus, per rung, a few
    integers per non-empty cell. Fine rungs below the typical spacing
    between points have roughly one cell per point; coarser rungs have far
    fewer. nbytes() reports the total.
 
    Picking the rungs for a query (levels_for):
        fine   = largest rung with side * sqrt(2) <= eps
        coarse = smallest rung with side >= eps
    Each can be up to 2x off the ideal size. That never affects
    correctness, only how well seeds are ordered and how soon phase 2 can
    stop. 
    """
 
    def __init__(self, xy, eps_min, eps_max):
        xy = np.asarray(xy, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) == 0:
            raise ValueError("xy must be a non-empty (n, 2) array")
        if not (0 < eps_min <= eps_max):
            raise ValueError("need 0 < eps_min <= eps_max")
        self.n = len(xy)
        self.eps_min = float(eps_min)
        self.eps_max = float(eps_max)
 
        # rung 0 is slightly under eps_min/sqrt(2), so eps_min selects it
        # even after the _TOL margin
        self.base = self.eps_min / SQRT2 / (1 + 4 * _TOL)
        top = 0
        while self.base * 2 ** top < self.eps_max * (1 + 4 * _TOL):
            top += 1
        self.sides = self.base * 2.0 ** np.arange(top + 1)
 
        # integer coordinates on the finest rung
        self.origin = xy.min(axis=0)
        ij = np.floor((xy - self.origin) / self.base)
        if ij.max() >= _MAX_CELLS_PER_AXIS:
            raise ValueError(
                f"eps_min={eps_min} is too small for data this wide: the "
                f"finest rung would need over {_MAX_CELLS_PER_AXIS:,} cells "
                f"per axis. Raise eps_min.")
        ij = ij.astype(np.int64)
 
        # one sort, shared by every rung
        idx_t = np.int32 if self.n < 2 ** 31 - 1 else np.int64
        key0 = morton(ij[:, 0], ij[:, 1])
        del ij
        order = np.argsort(key0, kind="stable")
        key0 = key0[order]
        self.point_order = order.astype(idx_t)
        del order
 
        self.levels = []
        for r in range(top + 1):
            k = key0 >> np.uint64(2 * r) if r else key0
            new_cell = np.empty(self.n, dtype=bool)
            new_cell[0] = True
            np.not_equal(k[1:], k[:-1], out=new_cell[1:])
            starts = np.flatnonzero(new_cell).astype(idx_t)
            del new_cell
            keys = k[starts]
            counts = np.diff(np.append(starts, self.n)).astype(idx_t)
            side = self.sides[r]
            self.levels.append(HierLevel(
                r, side, self.point_order, keys, starts, counts,
                as_fine=side * SQRT2 <= self.sides[-1] * (1 - _TOL),
                as_coarse=side >= self.sides[0] * SQRT2 * (1 + _TOL)))
            del k
 
    # choosing rungs 
    def fine_rung(self, eps):
        """Largest rung whose cells have diagonal <= eps, or None."""
        ok = np.flatnonzero(self.sides * SQRT2 <= eps * (1 - _TOL))
        return int(ok[-1]) if len(ok) else None
 
    def coarse_rung(self, eps):
        """Smallest rung whose cells are at least eps across, or None."""
        ok = np.flatnonzero(self.sides >= eps * (1 + _TOL))
        return int(ok[0]) if len(ok) else None
 
    @property
    def eps_range(self):
        """(lowest, highest) eps this index can serve.
 
        At least [eps_min, eps_max] as requested, and usually a little more
        at the top: the top rung can be up to 2x eps_max.
        """
        return (self.sides[0] * SQRT2 / (1 - _TOL),
                self.sides[-1] / (1 + _TOL))
 
    def levels_for(self, eps):
        """(fine, coarse) grids for this eps."""
        f, c = self.fine_rung(eps), self.coarse_rung(eps)
        if f is None or c is None:
            lo, hi = self.eps_range
            raise ValueError(
                f"eps={eps} is outside this index's range [{lo:.6g}, {hi:.6g}]. "
                f"Build it with a wider [eps_min, eps_max].")
        return self.levels[f], self.levels[c]
 
    # reporting 
    def nbytes(self):
        """Approximate memory held by the index, in bytes."""
        shared = self.point_order.nbytes
        return shared + sum(_level_nbytes(lv, shared_points=True)
                            for lv in self.levels)
 
    def describe(self):
        """One line per rung: side, cells, and densest cell."""
        rows = []
        for lv in self.levels:
            rows.append(f"  rung {lv.rung:>2}  side {lv.side:<12.6g} "
                        f"{lv.n_cells:>12,} cells   max count {int(lv.counts.max()):>8,}")
        return "\n".join(rows)
 
 
def _level_nbytes(lv, shared_points=False):
    arrays = [lv.cell_keys, lv.starts, lv.counts, lv.by_count,
              lv.block, lv.by_block]
    if not shared_points:
        arrays.append(lv.point_order)
    return sum(a.nbytes for a in arrays if a is not None)
 
 
class _Walk:
    """Per-query position in one grid's sorted cell array.
 
    `pos` only moves forward: once every point in a cell is labelled, the
    cell can never offer a seed again. Each visited cell also keeps a cursor
    to its first possibly-unlabelled point, which also only moves forward, so
    scanning for live points costs O(n) in total over a whole query.
 
    Cursors are created only for cells the query actually visits, so a
    top-k query touches index state in proportion to the cells it looks at,
    not to the size of the index.
    """
 
    def __init__(self, level, order, score, minPts, labels, unseen):
        self.level = level
        self.order = order
        self.score = score
        self.minPts = minPts
        self.labels = labels
        self.unseen = unseen
        self.pos = 0
        self.cursor = {}
 
    def live_point(self, c):
        """First unlabelled point in cell c, or -1."""
        lv = self.level
        cur = self.cursor.get(c)
        if cur is None:
            cur = int(lv.starts[c])
        end = int(lv.starts[c]) + int(lv.counts[c])
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