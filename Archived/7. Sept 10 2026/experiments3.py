"""
exp7_indexed_scaling.py
========================
Compares scaling (runtime + range queries) across all four variants:
  1. Vanilla DBSCAN            (no index,  no density-first)
  2. Vanilla DBSCAN + R-tree   (indexed,   no density-first)
  3. GDBSCAN (MaxRS)           (no index,  density-first)
  4. GDBSCAN (MaxRS) + R-tree  (indexed,   density-first)

All four are capped at the same k matching E2/E4/E5
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from benchmark_topk import DBSCAN, DBSCAN_Optimized, euclidean_distance
from benchmark_indexed import DBSCAN_indexed, DBSCAN_Optimized_indexed
from spatial_index import RTreeIndex

RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RUNS = 5


def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def save_csv(rows, headers, name):
    import csv
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  Saved: {path}")


def run_unindexed(data, eps, minPts, max_iterations, runs=RUNS):
    """Vanilla DBSCAN and GDBSCAN (MaxRS), no spatial index."""
    times_v, rq_v = [], []
    for _ in range(runs):
        t0 = time.time()
        _, _, rq = DBSCAN(data, euclidean_distance, eps, minPts,
                           max_iterations=max_iterations)
        times_v.append(time.time() - t0)
        rq_v.append(rq)

    times_o, rq_o = [], []
    for _ in range(runs):
        t0 = time.time()
        _, _, _, rq = DBSCAN_Optimized(data, euclidean_distance, eps, minPts,
                                        max_iterations=max_iterations)
        times_o.append(time.time() - t0)
        rq_o.append(rq)

    return (
        {"time_mean": np.mean(times_v), "rq_mean": np.mean(rq_v)},
        {"time_mean": np.mean(times_o), "rq_mean": np.mean(rq_o)},
    )


def run_indexed(data, eps, minPts, max_iterations, runs=RUNS):
    """
    Vanilla DBSCAN and GDBSCAN (MaxRS), both R-tree backed.

    The R-tree is built ONCE per dataset/subset and reused across all
    `runs` repetitions, excluding index-build time from the timing —
    matching how a real system would amortize index construction
    across many queries (see spatial_index.py / benchmark_indexed.py
    docstrings).
    """
    rtree = RTreeIndex(data)

    times_v, rq_v = [], []
    for _ in range(runs):
        t0 = time.time()
        _, _, rq = DBSCAN_indexed(data, euclidean_distance, eps, minPts,
                                   max_iterations=max_iterations, rtree=rtree)
        times_v.append(time.time() - t0)
        rq_v.append(rq)

    times_o, rq_o = [], []
    for _ in range(runs):
        t0 = time.time()
        _, _, _, rq = DBSCAN_Optimized_indexed(
            data, euclidean_distance, eps, minPts,
            max_iterations=max_iterations, rtree=rtree
        )
        times_o.append(time.time() - t0)
        rq_o.append(rq)

    return (
        {"time_mean": np.mean(times_v), "rq_mean": np.mean(rq_v)},
        {"time_mean": np.mean(times_o), "rq_mean": np.mean(rq_o)},
    )


def exp7_indexed_scaling(atlanta_data, eps=0.5, minPts=5, k=3):
    print("\n[E7] Indexed scaling experiment")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    n_full = len(atlanta_data)

    ns = []
    v_stats, vidx_stats, o_stats, oidx_stats = [], [], [], []
    rows = []

    for frac in fractions:
        n = max(50, int(n_full * frac))
        idx = np.random.choice(n_full, n, replace=False)
        subset = atlanta_data[idx]

        # sv, so = run_unindexed(subset, eps, minPts, max_iterations=k)
        svidx, soidx = run_indexed(subset, eps, minPts, max_iterations=k)

        ns.append(n)
        # v_stats.append(sv); 
        vidx_stats.append(svidx)
        # o_stats.append(so); 
        oidx_stats.append(soidx)

        rows.append([
            n, frac,
            # sv["time_mean"],    sv["rq_mean"],
            svidx["time_mean"], svidx["rq_mean"],
            # so["time_mean"],    so["rq_mean"],
            soidx["time_mean"], soidx["rq_mean"],
        ])
        print(f"  n={n}: "
              # f"vanilla t={sv['time_mean']:.4f}s rq={sv['rq_mean']:.0f} | "
              f"vanilla+rtree t={svidx['time_mean']:.4f}s rq={svidx['rq_mean']:.0f} | "
              # f"gdbscan t={so['time_mean']:.4f}s rq={so['rq_mean']:.0f} | "
              f"gdbscan+rtree t={soidx['time_mean']:.4f}s rq={soidx['rq_mean']:.0f}")

    save_csv(
        rows,
        ["n", "fraction",
         "vanilla_time",        "vanilla_rq",
         "vanilla_rtree_time",  "vanilla_rtree_rq",
         "gdbscan_time",        "gdbscan_rq",
         "gdbscan_rtree_time",  "gdbscan_rtree_rq"],
        "e7_indexed_scaling.csv",
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    series = [
        # ("Vanilla DBSCAN",            v_stats,    "o-",  "#2196F3"),
        ("Vanilla DBSCAN + R-tree",   vidx_stats, "o--", "#64B5F6"),
        # (f"GDBSCAN (MaxRS, k={k})",   o_stats,    "^-",  "#FF5722"),
        (f"GDBSCAN (MaxRS, k={k}) + R-tree", oidx_stats, "^--", "#FF9E80"),
    ]

    for label, stats, style, color in series:
        ax1.plot(ns, [s["time_mean"] for s in stats], style, label=label,
                  color=color, linewidth=2, markersize=7)
        ax2.plot(ns, [s["rq_mean"] for s in stats], style, label=label,
                  color=color, linewidth=2, markersize=7)

    ax1.set_xlabel("Number of points (n)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime vs. Dataset Size", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Number of points (n)", fontsize=12)
    ax2.set_ylabel("Range queries issued", fontsize=12)
    ax2.set_title("Range Queries vs. Dataset Size", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Indexed Scaling: Atlanta Restaurants (ε={eps} km, minPts={minPts}, k={k})",
        fontsize=13)
    plt.tight_layout()
    save_fig(fig, "e7_indexed_scaling.png")


if __name__ == "__main__":
    import numpy as np
    from benchmark_topk import generate_sparse_data

    print("Running exp7_indexed_scaling.py standalone on synthetic data...")
    np.random.seed(42)
    data = generate_sparse_data(n_points=1000, noise_ratio=0.7)
    exp7_indexed_scaling(data, eps=1.0, minPts=5, k=3)