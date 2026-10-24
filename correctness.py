"""
correctness.py
===============
checks a DBSCAN labelling against the definition of DBSCA

The reference is built directly from the definition, using scipy's KD-tree 

  core     |N_eps(p)| >= minPts, counting p itself and every duplicate

  cluster  a connected component of core points,  two core points
           are linked if within eps of each other

  border   not core, but within eps of at least one core point. 

  noise    everything else

A labelling passes if:
  1. each cluster's core points are exactly one whole component

  2. full run: every component is returned
     truncated run (k given): at most k components are returned

  3. every border point next to a returned cluster is labelled with one
     of the returned clusters it touches

Labels are an array indexed by point position: -1 for noise, any other
integer for a cluster id. For the older functions that return a
tuple-keyed dict, convert with labels_from_tuple_dict().

"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


class Reference:
    """Ground truth for one (data, eps, minPts)."""

    def __init__(self, xy, eps, minPts):
        xy = np.asarray(xy, dtype=float)
        self.n = len(xy)
        self.eps = eps
        self.minPts = minPts

        tree = cKDTree(xy)
        # neighbourhood sizes, including the point itself
        self.nbr_count = np.asarray(
            tree.query_ball_point(xy, eps, return_length=True))
        self.core = self.nbr_count >= minPts

        # connected components of core points
        self.comp = np.full(self.n, -1, dtype=np.int64)
        core_ids = np.flatnonzero(self.core)
        self.n_comp = 0
        self.border_options = {}  # point id -> set of component ids it touches

        if len(core_ids) == 0:
            return

        ctree = cKDTree(xy[core_ids])
        pairs = ctree.query_pairs(eps, output_type="ndarray")
        m = len(core_ids)
        graph = coo_matrix(
            (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(m, m))
        self.n_comp, lab = connected_components(graph, directed=False)
        self.comp[core_ids] = lab
        self.comp_size = np.bincount(lab, minlength=self.n_comp)

        # which clusters each non-core point could legally join
        noncore = np.flatnonzero(~self.core)
        if len(noncore):
            hits = ctree.query_ball_point(xy[noncore], eps)
            for p, lst in zip(noncore, hits):
                if lst:
                    self.border_options[int(p)] = set(lab[lst].tolist())


def validate(labels, xy, eps, minPts, k=None, ref=None):
    """
    Check a labelling. Returns (ok, problems, summary).

    k=None checks a full run; an integer checks a run truncated at k
    clusters. Pass a prebuilt Reference to reuse it across several
    labellings of the same data.
    """
    labels = np.asarray(labels, dtype=np.int64)
    if ref is None:
        ref = Reference(xy, eps, minPts)
    problems = []

    if len(labels) != ref.n:
        return False, [f"{len(labels)} labels for {ref.n} points"], {}

    core_ids = np.flatnonzero(ref.core)
    core_lab = labels[core_ids]
    core_comp = ref.comp[core_ids]
    in_cluster = core_lab != -1

    #  1. label <-> component must be a one-to-one match 
    pairs = np.unique(
        np.stack([core_lab[in_cluster], core_comp[in_cluster]], axis=1),
        axis=0) if in_cluster.any() else np.empty((0, 2), dtype=np.int64)

    labs, lab_n = np.unique(pairs[:, 0], return_counts=True)
    for L in labs[lab_n > 1]:
        problems.append(f"cluster {L} merges {lab_n[labs == L][0]} separate components")

    comps, comp_n = np.unique(pairs[:, 1], return_counts=True)
    for c in comps[comp_n > 1]:
        problems.append(f"component {c} is split across {comp_n[comps == c][0]} cluster labels")

    returned = set(comps.tolist())

    # every core point of a returned component must be in a cluster
    if len(returned):
        labelled_per_comp = np.bincount(
            core_comp[in_cluster], minlength=ref.n_comp)
        for c in returned:
            if labelled_per_comp[c] != ref.comp_size[c]:
                problems.append(
                    f"component {c}: {labelled_per_comp[c]} of "
                    f"{ref.comp_size[c]} core points labelled")

    # a label with no core point at all is not a DBSCAN cluster
    all_labs = np.unique(labels[labels != -1])
    for L in np.setdiff1d(all_labs, labs):
        problems.append(f"cluster {L} contains no core point")

    #  2. how many clusters 
    if k is None:
        if len(returned) != ref.n_comp:
            problems.append(
                f"full run returned {len(returned)} of {ref.n_comp} clusters")
    elif len(returned) > k:
        problems.append(f"asked for k={k}, returned {len(returned)} clusters")

    #  3. non-core points 
    comp_of_label = {int(L): int(c) for L, c in pairs}
    noncore = np.flatnonzero(~ref.core)
    bad_border = bad_noise = 0
    for p in noncore:
        L = int(labels[p])
        options = ref.border_options.get(int(p), set())
        reachable = options & returned
        if reachable:
            # a returned cluster touches it, so expansion must have claimed it
            if L == -1 or comp_of_label.get(L) not in options:
                bad_border += 1
                if bad_border <= 5:
                    problems.append(
                        f"point {p}: border of cluster(s) {sorted(reachable)}, "
                        f"labelled {L}")
        elif L != -1:
            bad_noise += 1
            if bad_noise <= 5:
                problems.append(f"point {p}: should be noise, labelled {L}")
    if bad_border > 5:
        problems.append(f"... {bad_border} bad border points in total")
    if bad_noise > 5:
        problems.append(f"... {bad_noise} bad noise points in total")

    summary = {
        "n": ref.n,
        "core": int(ref.core.sum()),
        "reference_clusters": int(ref.n_comp),
        "returned_clusters": len(returned),
        "noise": int((labels == -1).sum()),
    }
    return len(problems) == 0, problems, summary


def assert_valid(labels, xy, eps, minPts, k=None, ref=None):
    """validate(), raising AssertionError with the first problems found."""
    ok, problems, summary = validate(labels, xy, eps, minPts, k=k, ref=ref)
    if not ok:
        raise AssertionError(
            "labelling is not valid DBSCAN:\n  " + "\n  ".join(problems[:10]))
    return summary


def labels_from_tuple_dict(label_dict, xy):
    """Convert the older {tuple(point): label} output to a label array.

    Points at the same coordinates share one dict entry, so they all get
    the same label here. None (never visited) becomes -1.
    """
    out = np.empty(len(xy), dtype=np.int64)
    for i, p in enumerate(xy):
        v = label_dict.get(tuple(p))
        out[i] = -1 if v is None else v
    return out