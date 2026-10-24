"""
spatial_index.py
=================
R-tree wrapper with libspatialindex

"""

import numpy as np
from rtree import index as _rindex


class RTreeIndex:

    def __init__(self, points):
        """
        Parameters: points stored as tuples
        """
        self._points = [tuple(p) for p in points]
        self._arr = np.asarray(self._points, dtype=float)

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
        indices, so they count as two points. range_query returns tuples,
        and anything keyed on those tuples merges duplicates into one.
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

    def __len__(self):
        return len(self._points)


def build_index(points):
    return RTreeIndex(points)