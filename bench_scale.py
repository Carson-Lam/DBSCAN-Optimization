"""
bench_scale.py
===============
End-to-end scale check on synthetic data. Run before a long real-data run
to confirm the machine has the memory and to see the rough speeds:

    python bench_scale.py 30000000          # n points
    python bench_scale.py 30000000 300      # let the baseline run 300 s

Builds the Indexicon spatial index and the hierarchical grid, runs top-1/3/5
queries, then runs full indexed DBSCAN (the baseline) for a time budget and
extrapolates how long it would take to finish.

Data: 5 clusters x 300 points plus uniform noise at 10 points per unit^2,
so the top-k answer is the same at every n.
"""

import resource
import sys
import time

import numpy as np

from algorithms import DBSCAN_grid, DBSCAN_indexed
from bench_hgrid import fixed_clusters
from grid_index import HierarchicalGridIndex
from spatial_index import IndexiconIndex

EPS, MINPTS = 0.30, 5


def peak_gb():
    # ru_maxrss is KB on Linux, bytes on macOS
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**30 if sys.platform == "darwin" else r / 2**20


def main(n, budget):
    t0 = time.time()
    xy = fixed_clusters(n)
    print(f"[{n:,} points] generated in {time.time() - t0:.1f}s, peak {peak_gb():.2f} GB", flush=True)

    t0 = time.time()
    sp = IndexiconIndex(xy)
    print(f"Indexicon KD-tree: {time.time() - t0:7.1f}s  {sp.nbytes() / 2**30:.2f} GB  "
          f"peak {peak_gb():.2f} GB", flush=True)

    t0 = time.time()
    H = HierarchicalGridIndex(xy, eps_min=0.2, eps_max=0.6)
    print(f"hierarchical grid: {time.time() - t0:7.1f}s  {H.nbytes() / 2**30:.2f} GB  "
          f"{len(H.sides)} rungs  peak {peak_gb():.2f} GB", flush=True)

    for k in (1, 3, 5):
        t0 = time.time()
        lab, _, _, rq = DBSCAN_grid(xy, EPS, MINPTS, max_iterations=k,
                                        tree=sp, grid=H, refine=8)
        sizes = sorted(np.bincount(lab[lab > 0]).tolist(), reverse=True)[:k]
        print(f"top-{k}: {time.time() - t0:6.2f}s  {rq:,} range queries  "
              f"cluster sizes {sizes}  peak {peak_gb():.2f} GB", flush=True)

    class _Stop(Exception):
        pass

    state = {"t0": time.time()}

    def progress(done, total, clusters, rq):
        state.update(done=done, rq=rq, el=time.time() - state["t0"])
        if state["el"] > budget:
            raise _Stop

    try:
        _, _, rq = DBSCAN_indexed(xy, EPS, MINPTS, tree=sp, progress=progress)
        print(f"baseline finished: {time.time() - state['t0']:.1f}s, {rq:,} range queries", flush=True)
    except _Stop:
        rate = state["rq"] / state["el"]
        print(f"baseline stopped after {state['el']:.0f}s: {state['rq']:,} range queries "
              f"({rate:,.0f}/s). It needs about {n:,} -> est. {n / rate / 60:.0f} min total. "
              f"peak {peak_gb():.2f} GB", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000,
         float(sys.argv[2]) if len(sys.argv) > 2 else 60)