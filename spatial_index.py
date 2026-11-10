"""
spatial_index.py
=================
Spatial indexes that answer DBSCAN's eps-range queries.

  IndexiconIndex   Default. Backed by Indexicon (Simatis, Bouros, Mamoulis,
                   SIGSPATIAL 2026), a header-only C++ library with Python
                   bindings: pip install indexicon. Uses rtree by default 
                   or r-trees. Memory about 40 bytes per point

  RTreeIndex       A libspatialindex wrapper. Kept for backward compatibility.
                   It stores every point as a Python tuple, which takes up
                   more spacae than IndexiconIndex. Memory about 100 bytes per point

Both answer an eps-range query the same way: a box query for the square
[x-eps, x+eps] x [y-eps, y+eps] (inclusive edges), then an exact distance
filter dx^2 + dy^2 <= eps^2. 

The id-based DBSCAN functions call neighbors_of(i, eps): "every point within
eps of point i", by point index. Point i itself is included, and points at
identical coordinates are separate points.
"""

import numpy as np


class IndexiconIndex:
    """Exact eps-range queries over a fixed 2-D point set, using Indexicon.

    Parameters
    ----------
    xy : (n, 2) array of coordinates, in a planar unit (e.g. km). 

    kind : "kdtree" (default) or "rtree". Identical results but kdtree is slightly faster
    """

    def __init__(self, xy, kind="rtree"):
        import indexicon  # imported here so RTreeIndex works without it

        xy = np.ascontiguousarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must be an (n, 2) array")
        self.xy = xy
        self.n = len(xy)
        self.kind = kind
        if kind == "kdtree":
            self.tree = indexicon.KDTree(dims=2)
        elif kind == "rtree":
            self.tree = indexicon.RTree(data="point", dims=2)
        else:
            raise ValueError('kind must be "kdtree" or "rtree"')
        if self.n:
            self.tree.bulk_load(xy)          # ids default to 0..n-1
        # reused query-box buffers: avoids two array allocations per query
        self._lo = np.empty(2)
        self._hi = np.empty(2)

    def neighbors_of(self, i, eps):
        """Ids of all points within distance eps of point i (inclusive),
        including i itself. Returns a list of Python ints."""
        x, y = self.xy[i]
        return self._query(float(x), float(y), eps)

    def range_query_ids(self, q, eps):
        """Ids of all points within distance eps of coordinates q."""
        return self._query(float(q[0]), float(q[1]), eps)

    def _query(self, x, y, eps):
        # Pad the box by a few thousand ulps so float rounding in x +/- eps
        # can never drop a point the exact distance test below would keep.
        pad = (abs(x) + abs(y) + eps) * 1e-12
        r = eps + pad
        lo, hi = self._lo, self._hi
        lo[0] = x - r; lo[1] = y - r
        hi[0] = x + r; hi[1] = y + r
        ids = self.tree.query(lo, hi)
        if len(ids) == 0:
            return []
        d = self.xy[ids]
        dx = d[:, 0] - x
        dy = d[:, 1] - y
        return ids[dx * dx + dy * dy <= eps * eps].tolist()

    def nbytes(self):
        """Approximate memory: the coordinate array plus the C++ tree."""
        return self.xy.nbytes + int(self.tree.statistics()["size_bytes"])

    def __len__(self):
        return self.n


class RTreeIndex:
    """The original libspatialindex wrapper (legacy)."""

    def __init__(self, points):
        """
        Parameters: points stored as tuples
        """
        from rtree import index as _rindex  # only needed for this class

        self._points = [tuple(p) for p in points]
        self._arr = np.asarray(self._points, dtype=float)
        self.n = len(self._points)

        p = _rindex.Property()
        p.dimension = 2

        def _gen():
            for i, (x, y) in enumerate(self._points):
                yield (i, (x, y, x, y), None)

        self.idx = _rindex.Index(_gen(), properties=p)

    def range_query(self, q, eps):
        """
        Same RangeQuery as in DBSCAN
        """
        qx, qy = q[0], q[1]
        # Step 1: Bounding box
        bbox = (qx - eps, qy - eps, qx + eps, qy + eps)
        candidate_ids = self.idx.intersection(bbox)

        # Step 2: Points withinn circle of bounding box
        out = []
        eps2 = eps * eps
        for i in candidate_ids:
            px, py = self._points[i]
            dx = px - qx
            dy = py - qy
            if dx * dx + dy * dy <= eps2:
                out.append(self._points[i])
        return out

    def range_query_ids(self, q, eps):
        """
        Like range_query, but returns point indices instead of coordinate
        tuples. Indices follow the order the points were passed in.

        Two points at the same coordinates come back as two separate
        indices so they count as two points. 
        """
        qx, qy = q[0], q[1]
        bbox = (qx - eps, qy - eps, qx + eps, qy + eps)
        eps2 = eps * eps
        pts = self._points
        out = []
        for i in self.idx.intersection(bbox):
            px, py = pts[i]
            dx = px - qx
            dy = py - qy
            if dx * dx + dy * dy <= eps2:
                out.append(i)
        return out

    def neighbors_of(self, i, eps):
        """Ids of all points within eps of point i (same interface as
        IndexiconIndex)."""
        return self.range_query_ids(self._points[i], eps)

    def __len__(self):
        return len(self._points)


def build_index(points, kind="kdtree"):
    """Default spatial index for the id-based DBSCAN functions."""
    return IndexiconIndex(points, kind=kind)