"""
experiments.py
==============
Reproduces all experimental results for the paper:
  "Result-Sensitive DBSCAN via Maximum Range Sum"

Run this file once to produce every figure (PNG) and table (CSV)
referenced in the paper. All outputs are saved to ./results/.

Dependencies: numpy, matplotlib, scikit-learn, requests


E2 Scaling: 3 algos
one completely vanilla full dbscan (should just be linear)
MaxRS DBSCAN (with k)
one completely vanilla dbscan (with k)

k3
Is there any cherrypicking?
Make legend, labels larger

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import json
import os
import csv

# ── Project imports ──────────────────────────────────────────────────────────
# Assumes benchmark.py and osm_cluster.py are in the same directory.
from benchmark_topk import (
    DBSCAN,
    DBSCAN_Optimized,
    euclidean_distance,
    generate_sparse_data,
    generate_dense_data,
    generate_varied_density_data,
)
from osm_cluster import load_osm_data, convert_to_xy

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUNS = 5  # Repetitions per timing measurement


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def load_osm_city(json_file, bbox):
    """Load an OSM restaurant JSON file and convert to km coordinates."""
    restaurants = load_osm_data(json_file)
    if restaurants is None:
        return None
    data = convert_to_xy(restaurants, bbox)
    print(f"  Loaded {len(data)} points from {json_file}")
    return data


def load_all_real_datasets():
    """
    Load Atlanta, NYC, and Chicago restaurant datasets.
    Run osm.py with the bounding boxes below to generate the JSON files.
    """
    datasets = {}

    atlanta_bbox = (33.6490, -84.5510, 33.8860, -84.2890)
    nyc_bbox     = (40.4774, -74.2591, 40.9176, -73.7004)
    chicago_bbox = (41.6445, -87.9401, 42.0230, -87.5240)

    specs = [
        ("Atlanta",  "atlanta_restaurants_osm.json",  atlanta_bbox),
        ("NYC",      "nyc_restaurants_osm.json",      nyc_bbox),
        ("Chicago",  "chicago_restaurants_osm.json",  chicago_bbox),
    ]

    for name, fname, bbox in specs:
        if os.path.exists(fname):
            data = load_osm_city(fname, bbox)
            if data is not None:
                datasets[name] = data
        else:
            print(f"  [SKIP] {fname} not found. Run osm.py with bbox={bbox}")

    return datasets


def make_synthetic_datasets():
    """
    Synthetic datasets with known ground truth.
    Returns dict of {name: (data, n_true_clusters)}.
    """
    return {
        "Sparse (70% noise)":       (generate_sparse_data(n_points=800,  noise_ratio=0.70), 3),
        "Sparse (80% noise)":       (generate_sparse_data(n_points=1000, noise_ratio=0.80), 3),
        "Dense (10 clusters)":      (generate_dense_data(n_points=1000,  n_clusters=10),   10),
        "Varied density (5 clust)": (generate_varied_density_data(n_points=1000),           5),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_both(data, eps, minPts, max_iterations=None, runs=RUNS):
    """
    Run vanilla DBSCAN and DBSCAN-MaxRS, return averaged stats.
    Returns two dicts: stats_orig, stats_opt.
    """
    # Vanilla
    times_v, rq_v = [], []
    for _ in range(runs):
        t0 = time.time()
        labels_v, _, rq = DBSCAN(data, euclidean_distance, eps, minPts,
                                  max_iterations=max_iterations)
        times_v.append(time.time() - t0)
        rq_v.append(rq)

    # Optimized
    times_o, rq_o, maxrs_o = [], [], []
    for _ in range(runs):
        t0 = time.time()
        labels_o, _, mc, rq = DBSCAN_Optimized(data, euclidean_distance, eps, minPts,
                                                max_iterations=max_iterations)
        times_o.append(time.time() - t0)
        rq_o.append(rq)
        maxrs_o.append(mc)

    def cluster_sizes(labels):
        ids = set(labels.values()) - {-1, None}
        return sorted([sum(1 for v in labels.values() if v == i) for i in ids], reverse=True)

    stats_v = {
        "time_mean": np.mean(times_v),
        "time_std":  np.std(times_v),
        "rq_mean":   np.mean(rq_v),
        "labels":    labels_v,
        "cluster_sizes": cluster_sizes(labels_v),
        "n_clusters": len(set(labels_v.values()) - {-1, None}),
        "n_noise":    sum(1 for v in labels_v.values() if v == -1),
    }
    stats_o = {
        "time_mean":  np.mean(times_o),
        "time_std":   np.std(times_o),
        "rq_mean":    np.mean(rq_o),
        "maxrs_mean": np.mean(maxrs_o),
        "labels":     labels_o,
        "cluster_sizes": cluster_sizes(labels_o),
        "n_clusters": len(set(labels_o.values()) - {-1, None}),
        "n_noise":    sum(1 for v in labels_o.values() if v == -1),
    }
    return stats_v, stats_o

def run_vanilla_full(data, eps, minPts, runs=RUNS):
    """Run vanilla DBSCAN to full completion, no k cap."""
    times_v, rq_v = [], []
    for _ in range(runs):
        t0 = time.time()
        labels_v, _, rq = DBSCAN(data, euclidean_distance, eps, minPts,
                                  max_iterations=None)
        times_v.append(time.time() - t0)
        rq_v.append(rq)
    return {
        "time_mean": np.mean(times_v),
        "time_std":  np.std(times_v),
        "rq_mean":   np.mean(rq_v),
    }

def save_fig(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def save_csv(rows, headers, name):
    path = os.path.join(RESULTS_DIR, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — CORRECTNESS
# Run both algorithms to COMPLETION (no max_iterations cap) on synthetic data
# with known structure. Verify cluster assignments are identical.
# ═══════════════════════════════════════════════════════════════════════════════

def exp1_correctness():
    print("\n[E1] Correctness verification")
    synth = make_synthetic_datasets()
    rows = []
    all_pass = True

    for name, (data, n_true) in synth.items():
        eps, minPts = 1.0, 5
        sv, so = run_both(data, eps, minPts, max_iterations=None, runs=1)

        lv = sv["labels"]
        lo = so["labels"]

        # Build canonical label mapping: sort clusters by size desc,
        # then check every point falls in the same canonical cluster.
        # Simpler check: n_clusters must match and every cluster found
        # by vanilla must appear in optimized with the same point set.
        v_clusters = {}
        for pt, c in lv.items():
            if c not in (-1, None):
                v_clusters.setdefault(c, set()).add(pt)

        o_clusters = {}
        for pt, c in lo.items():
            if c not in (-1, None):
                o_clusters.setdefault(c, set()).add(pt)

        v_sets = sorted([frozenset(s) for s in v_clusters.values()], key=len, reverse=True)
        o_sets = sorted([frozenset(s) for s in o_clusters.values()], key=len, reverse=True)

        match = (v_sets == o_sets)
        if not match:
            all_pass = False

        status = "PASS" if match else "FAIL"
        rows.append([name, n_true, sv["n_clusters"], so["n_clusters"], status])
        print(f"  {name}: vanilla={sv['n_clusters']} clusters, "
              f"optimized={so['n_clusters']} clusters → {status}")

    save_csv(rows,
             ["Dataset", "True clusters", "Vanilla clusters", "Optimized clusters", "Match"],
             "e1_correctness.csv")
    print(f"  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILURES ✗'}")
    return all_pass


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — SCALING (runtime and range queries vs. n)
# Subsample Atlanta data at 10/25/50/75/100% and measure both algorithms.
# ═══════════════════════════════════════════════════════════════════════════════

def exp2_scaling(atlanta_data, eps=0.5, minPts=5, k=3):
    print("\n[E2] Scaling experiment")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    n_full = len(atlanta_data)

    stats_full, stats_vk, stats_ok, ns, rows = [], [], [], [], []
    
    for frac in fractions:
        n = max(50, int(n_full * frac))
        idx = np.random.choice(n_full, n, replace=False)
        subset = atlanta_data[idx]

        sf = run_vanilla_full(subset, eps, minPts)
        sv, so = run_both(subset, eps, minPts, max_iterations=k)

        ns.append(n)
        stats_full.append(sf)
        stats_vk.append(sv)
        stats_ok.append(so)

        rows.append([n, frac,
                     sf["rq_mean"], sf["time_mean"],
                     sv["rq_mean"], sv["time_mean"],
                     so["rq_mean"], so["time_mean"]])
        print(f"  n={n}: vanilla_full rq={sf['rq_mean']:.0f}, "
              f"vanilla_k rq={sv['rq_mean']:.0f}, "
              f"gdbscan_k rq={so['rq_mean']:.0f}")

    save_csv(rows,
             ["n", "fraction",
              "vanilla_full_rq", "vanilla_full_time",
              "vanilla_k_rq",    "vanilla_k_time",
              "gdbscan_k_rq",    "gdbscan_k_time"],
             "e2_scaling.csv")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Runtime
    ax1.plot(ns, [s["time_mean"] for s in stats_full], "o-",
             label="Vanilla DBSCAN (full, no k)", color="#2196F3", linewidth=2, markersize=8)
    ax1.plot(ns, [s["time_mean"] for s in stats_vk], "s--",
             label=f"Vanilla DBSCAN (top-k, k={k})", color="#9C27B0", linewidth=2, markersize=8)
    ax1.plot(ns, [s["time_mean"] for s in stats_ok], "^-",
             label=f"GDBSCAN (top-k, k={k})", color="#FF5722", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of points (n)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime vs. Dataset Size", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Range queries
    ax2.plot(ns, [s["rq_mean"] for s in stats_full], "o-",
             label="Vanilla DBSCAN (full, no k)", color="#2196F3", linewidth=2, markersize=8)
    ax2.plot(ns, [s["rq_mean"] for s in stats_vk], "s--",
             label=f"Vanilla DBSCAN (top-k, k={k})", color="#9C27B0", linewidth=2, markersize=8)
    ax2.plot(ns, [s["rq_mean"] for s in stats_ok], "^-",
             label=f"GDBSCAN (top-k, k={k})", color="#FF5722", linewidth=2, markersize=8)
    ax2.set_xlabel("Number of points (n)", fontsize=12)
    ax2.set_ylabel("Range queries issued", fontsize=12)
    ax2.set_title("Range Queries vs. Dataset Size", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Scaling — Atlanta Restaurants (ε={eps} km, minPts={minPts}, k={k})",
        fontsize=13)
    plt.tight_layout()
    save_fig(fig, "e2_scaling.png")

# def exp2_scaling(atlanta_data, eps=0.5, minPts=5, k=5):
#     print("\n[E2] Scaling experiment")
#     fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
#     n_full = len(atlanta_data)

#     ns, times_v, times_o, rqs_v, rqs_o = [], [], [], [], []
#     rows = []

#     for frac in fractions:
#         n = max(50, int(n_full * frac))
#         idx = np.random.choice(n_full, n, replace=False)
#         subset = atlanta_data[idx]

#         sv, so = run_both(subset, eps, minPts, max_iterations=k)

#         ns.append(n)
#         times_v.append(sv["time_mean"])
#         times_o.append(so["time_mean"])
#         rqs_v.append(sv["rq_mean"])
#         rqs_o.append(so["rq_mean"])

#         rows.append([n, frac, sv["time_mean"], sv["time_std"],
#                      so["time_mean"], so["time_std"],
#                      sv["rq_mean"], so["rq_mean"]])
#         print(f"  n={n}: vanilla={sv['time_mean']:.3f}s, optimized={so['time_mean']:.3f}s, "
#               f"RQ vanilla={sv['rq_mean']:.0f}, RQ opt={so['rq_mean']:.0f}")

#     save_csv(rows,
#              ["n", "fraction", "vanilla_time", "vanilla_std",
#               "opt_time", "opt_std", "vanilla_rq", "opt_rq"],
#              "e2_scaling.csv")

#     # Plot
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

#     ax1.plot(ns, times_v, "o-", label="Vanilla DBSCAN", color="#2196F3")
#     ax1.plot(ns, times_o, "s-", label="DBSCAN-MaxRS", color="#FF5722")
#     ax1.set_xlabel("Number of points (n)")
#     ax1.set_ylabel("Runtime (seconds)")
#     ax1.set_title("Runtime vs. Dataset Size")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     ax2.plot(ns, rqs_v, "o-", label="Vanilla DBSCAN", color="#2196F3")
#     ax2.plot(ns, rqs_o, "s-", label="DBSCAN-MaxRS", color="#FF5722")
#     ax2.set_xlabel("Number of points (n)")
#     ax2.set_ylabel("Range queries issued")
#     ax2.set_title("Range Queries vs. Dataset Size")
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)

#     fig.suptitle(f"Scaling Experiment — Atlanta Restaurants (eps={eps}, minPts={minPts}, k={k})",
#                  fontsize=12)
#     plt.tight_layout()
#     save_fig(fig, "e2_scaling.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — TOP-K QUALITY
# Vary k from 1..10. Show cluster sizes found by each algorithm.
# Key result: vanilla finds arbitrary clusters, optimized finds the densest.
# ═══════════════════════════════════════════════════════════════════════════════

# def exp3_topk_quality(atlanta_data, eps=0.5, minPts=5):
#     print("\n[E3] Top-k quality experiment")
#     ks = list(range(1, 11))
#     rows = []

#     vanilla_total_pts = []
#     opt_total_pts     = []

#     for k in ks:
#         sv, so = run_both(atlanta_data, eps, minPts, max_iterations=k, runs=3)

#         v_total = sum(sv["cluster_sizes"])
#         o_total = sum(so["cluster_sizes"])
#         vanilla_total_pts.append(v_total)
#         opt_total_pts.append(o_total)

#         v_sizes_str = str(sv["cluster_sizes"])
#         o_sizes_str = str(so["cluster_sizes"])
#         rows.append([k, sv["n_clusters"], v_total, v_sizes_str,
#                         so["n_clusters"], o_total, o_sizes_str])
#         print(f"  k={k}: vanilla total_pts={v_total} {sv['cluster_sizes']}, "
#               f"opt total_pts={o_total} {so['cluster_sizes']}")

#     save_csv(rows,
#              ["k", "vanilla_n_clusters", "vanilla_total_pts", "vanilla_sizes",
#               "opt_n_clusters", "opt_total_pts", "opt_sizes"],
#              "e3_topk_quality.csv")

#     # Plot: total clustered points vs. k
#     fig, ax = plt.subplots(figsize=(8, 5))
#     ax.plot(ks, vanilla_total_pts, "o-", label="Vanilla DBSCAN (random order)", color="#2196F3")
#     ax.plot(ks, opt_total_pts,     "s-", label="DBSCAN-MaxRS (density-first)",   color="#FF5722")
#     ax.set_xlabel("k  (number of clusters requested)")
#     ax.set_ylabel("Total points in top-k clusters")
#     ax.set_title("Top-k Quality: Points Captured vs. k\n(higher = denser clusters found)")
#     ax.legend()
#     ax.grid(True, alpha=0.3)
#     plt.tight_layout()
#     save_fig(fig, "e3_topk_quality.png")

def exp3_topk_quality(atlanta_data, eps=0.5, minPts=5):
    print("\n[E3] Top-k quality experiment")
    ks = list(range(1, 11))
    rows = []

    vanilla_full_pts = []
    vanilla_k_pts    = []
    opt_k_pts        = []

    # Count total points in all clusters for vanilla full
    labels_full, _, _ = DBSCAN(atlanta_data, euclidean_distance, eps, minPts,
                                max_iterations=None)
    
    all_cluster_sizes = sorted(
        [sum(1 for v in labels_full.values() if v == i)
         for i in set(labels_full.values()) - {-1, None}],
        reverse=True
    )

    for k in ks:
        sv, so = run_both(atlanta_data, eps, minPts, max_iterations=k, runs=3)

        # Vanilla full: sum of top-k cluster sizes by size order
        vf_total = sum(all_cluster_sizes[:k])
        v_total  = sum(sv["cluster_sizes"])
        o_total  = sum(so["cluster_sizes"])

        vanilla_full_pts.append(vf_total)
        vanilla_k_pts.append(v_total)
        opt_k_pts.append(o_total)

        rows.append([k,
                     vf_total,
                     sv["n_clusters"], v_total, str(sv["cluster_sizes"]),
                     so["n_clusters"], o_total, str(so["cluster_sizes"])])
        print(f"  k={k}: vanilla_full={vf_total}, "
              f"vanilla_k={v_total} {sv['cluster_sizes']}, "
              f"gdbscan={o_total} {so['cluster_sizes']}")

    save_csv(rows,
             ["k", "vanilla_full_top_k_pts",
              "vanilla_k_clusters", "vanilla_k_total_pts", "vanilla_k_sizes",
              "gdbscan_clusters",   "gdbscan_total_pts",   "gdbscan_sizes"],
             "e3_topk_quality.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(ks, vanilla_full_pts, "o-",
            label="Vanilla DBSCAN (full run, top-k by size)", color="#2196F3",
            linewidth=2, markersize=8)
    ax.plot(ks, vanilla_k_pts, "s--",
            label="Vanilla DBSCAN (stopped at k)", color="#9C27B0",
            linewidth=2, markersize=8)
    ax.plot(ks, opt_k_pts, "^-",
            label="GDBSCAN (density-first, stopped at k)", color="#FF5722",
            linewidth=2, markersize=8)
    ax.set_xlabel("k  (number of clusters requested)", fontsize=12)
    ax.set_ylabel("Total points in top-k clusters", fontsize=12)
    ax.set_title(
        "Top-k Quality: Points Captured vs. k\n"
        "(higher = denser clusters found earlier)",
        fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_fig(fig, "e3_topk_quality.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4 — PARAMETER SENSITIVITY
# Grid over eps and minPts. Report runtime and range queries.
# ═══════════════════════════════════════════════════════════════════════════════

def exp4_parameter_sensitivity(atlanta_data, k=3):
    print("\n[E4] Parameter sensitivity")
    eps_vals    = [0.3, 0.5, 0.7, 1.0]
    minpts_vals = [3, 5, 8, 10]

    # Grids: rows=eps, cols=minPts
    grid_rq_vanilla = np.zeros((len(eps_vals), len(minpts_vals)))
    grid_rq_opt     = np.zeros((len(eps_vals), len(minpts_vals)))
    grid_time_vanilla = np.zeros((len(eps_vals), len(minpts_vals)))
    grid_time_opt     = np.zeros((len(eps_vals), len(minpts_vals)))

    rows = []
    for i, eps in enumerate(eps_vals):
        for j, minPts in enumerate(minpts_vals):
            sv, so = run_both(atlanta_data, eps, minPts, max_iterations=k, runs=3)
            grid_rq_vanilla[i, j]   = sv["rq_mean"]
            grid_rq_opt[i, j]       = so["rq_mean"]
            grid_time_vanilla[i, j] = sv["time_mean"]
            grid_time_opt[i, j]     = so["time_mean"]
            rows.append([eps, minPts, sv["time_mean"], so["time_mean"],
                         sv["rq_mean"], so["rq_mean"]])
            print(f"  eps={eps}, minPts={minPts}: "
                  f"vanilla rq={sv['rq_mean']:.0f}, opt rq={so['rq_mean']:.0f}")

    save_csv(rows,
             ["eps", "minPts", "vanilla_time", "opt_time", "vanilla_rq", "opt_rq"],
             "e4_sensitivity.csv")

    # Heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    titles = ["Vanilla DBSCAN — Range Queries",
              "DBSCAN-MaxRS — Range Queries",
              "Vanilla DBSCAN — Runtime (s)",
              "DBSCAN-MaxRS — Runtime (s)"]
    grids  = [grid_rq_vanilla, grid_rq_opt, grid_time_vanilla, grid_time_opt]

    for ax, title, grid in zip(axes.flat, titles, grids):
        im = ax.imshow(grid, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(minpts_vals)))
        ax.set_xticklabels([f"minPts={m}" for m in minpts_vals])
        ax.set_yticks(range(len(eps_vals)))
        ax.set_yticklabels([f"ε={e}" for e in eps_vals])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
        for ii in range(len(eps_vals)):
            for jj in range(len(minpts_vals)):
                ax.text(jj, ii, f"{grid[ii,jj]:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    fig.suptitle(f"Parameter Sensitivity — Atlanta Restaurants (k={k})", fontsize=13)
    plt.tight_layout()
    save_fig(fig, "e4_sensitivity_heatmaps.png")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5 — MULTI-DATASET COMPARISON
# Run both algorithms on all datasets (real + synthetic). Summary table.
# ═══════════════════════════════════════════════════════════════════════════════

def exp5_multi_dataset(real_datasets, eps=0.5, minPts=5, k=5):
    print("\n[E5] Multi-dataset comparison")
    rows = []
    all_datasets = {}

    # Real datasets use km-based eps
    for name, data in real_datasets.items():
        all_datasets[name] = (data, 0.5, 5)

    # Synthetic datasets use unit-space eps
    synth = make_synthetic_datasets()
    for name, (data, _) in synth.items():
        all_datasets[name] = (data, 1.0, 5)

    for name, (data, e, mp) in all_datasets.items():
        sv, so = run_both(data, e, mp, max_iterations=k)
        speedup = sv["time_mean"] / so["time_mean"] if so["time_mean"] > 0 else float("inf")
        rq_reduction = (1 - so["rq_mean"] / sv["rq_mean"]) * 100 if sv["rq_mean"] > 0 else 0.0

        rows.append([
            name, len(data), e, mp, k,
            f"{sv['time_mean']:.4f}", f"{so['time_mean']:.4f}",
            f"{speedup:.2f}x",
            f"{sv['rq_mean']:.0f}", f"{so['rq_mean']:.0f}",
            f"{rq_reduction:.1f}%",
            sv["n_clusters"], so["n_clusters"],
        ])
        print(f"  {name}: n={len(data)}, speedup={speedup:.2f}x, "
              f"RQ reduction={rq_reduction:.1f}%")

    save_csv(rows,
             ["Dataset", "n", "eps", "minPts", "k",
              "Vanilla time (s)", "Opt time (s)", "Speedup",
              "Vanilla RQ", "Opt RQ", "RQ reduction",
              "Vanilla clusters", "Opt clusters"],
             "e5_multi_dataset.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 6 — EARLY TERMINATION SAVINGS
# Show what % of range queries and points are skipped by early exit.
# Run on sparse synthetic datasets where noise dominates.
# ═══════════════════════════════════════════════════════════════════════════════

def exp6_early_termination():
    print("\n[E6] Early termination savings")
    noise_ratios = [0.50, 0.60, 0.70, 0.80, 0.90]
    eps, minPts, k = 1.0, 5, 3
    rows = []

    rq_savings  = []
    pts_savings = []
    labels_nrs  = []

    for nr in noise_ratios:
        data = generate_sparse_data(n_points=1000, noise_ratio=nr)
        sv, so = run_both(data, eps, minPts, max_iterations=k, runs=3)

        rq_save  = (1 - so["rq_mean"] / sv["rq_mean"]) * 100 if sv["rq_mean"] > 0 else 0.0
        pts_save = so["n_noise"] - sv["n_noise"]   # extra noise points avoided

        # Count unlabeled points when opt terminates early
        # (points marked noise by early exit, not by explicit range query)
        rq_savings.append(rq_save)
        pts_savings.append(so["n_noise"])
        labels_nrs.append(nr)

        rows.append([nr, sv["rq_mean"], so["rq_mean"], rq_save,
                     sv["n_noise"], so["n_noise"]])
        print(f"  noise={nr:.0%}: RQ savings={rq_save:.1f}%, "
              f"vanilla noise pts={sv['n_noise']}, opt noise pts={so['n_noise']}")

    save_csv(rows,
             ["noise_ratio", "vanilla_rq", "opt_rq", "rq_savings_pct",
              "vanilla_noise_pts", "opt_noise_pts"],
             "e6_early_termination.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([f"{nr:.0%}" for nr in noise_ratios], rq_savings, color="#FF5722", alpha=0.8)
    ax.set_xlabel("Noise ratio in dataset")
    ax.set_ylabel("Range query savings (%)")
    ax.set_title("Early Termination Benefit vs. Noise Ratio\n"
                 "(higher noise → more queries saved by early exit)")
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis="y")
    for i, v in enumerate(rq_savings):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", fontsize=9)
    plt.tight_layout()
    save_fig(fig, "e6_early_termination.png")


# ═══════════════════════════════════════════════════════════════════════════════
# BONUS — CLUSTER VISUALIZATION (reproduces the OSM figure in the paper)
# ═══════════════════════════════════════════════════════════════════════════════

def plot_clustering_comparison(data, eps, minPts, k, dataset_name, filename):
    """Side-by-side cluster visualization for a real dataset."""
    sv, so = run_both(data, eps, minPts, max_iterations=k, runs=1)
    lv, lo = sv["labels"], so["labels"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    def _plot(ax, labels, title):
        clusters = sorted(set(labels.values()) - {-1, None})
        colors = plt.cm.tab10(range(len(clusters)))
        for i, cid in enumerate(clusters):
            pts = np.array([p for p in data if labels[tuple(p)] == cid])
            ax.scatter(pts[:, 0], pts[:, 1], c=[colors[i]],
                       label=f"Cluster {cid} ({len(pts)} pts)",
                       edgecolor="k", s=60, alpha=0.75)
        noise = np.array([p for p in data if labels[tuple(p)] == -1])
        if len(noise):
            ax.scatter(noise[:, 0], noise[:, 1], c="lightgray",
                       marker="x", s=30, alpha=0.4, label=f"Noise ({len(noise)} pts)")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("X (km)"); ax.set_ylabel("Y (km)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    _plot(ax1, lv, f"Vanilla DBSCAN (random order)\nTop-{k} clusters found")
    _plot(ax2, lo, f"DBSCAN-MaxRS (density-first)\nTop-{k} clusters found")
    fig.suptitle(f"{dataset_name} — ε={eps} km, minPts={minPts}, k={k}", fontsize=13)
    plt.tight_layout()
    save_fig(fig, filename)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    np.random.seed(42)
    print("=" * 60)
    print("DBSCAN-MaxRS — Full Experimental Suite")
    print("=" * 60)

    # ── Load datasets ────────────────────────────────────────────
    print("\nLoading real-world datasets...")
    real_datasets = load_all_real_datasets()

    if not real_datasets:
        print("  No real datasets found. Run osm.py for Atlanta, NYC, Chicago first.")
        print("  Continuing with synthetic-only experiments.\n")

    # ── E1: Correctness ──────────────────────────────────────────
    exp1_correctness()

    # ── E2: Scaling ──────────────────────────────────────────────
    if "Atlanta" in real_datasets:
        exp2_scaling(real_datasets["Atlanta"], eps=0.5, minPts=5, k=5)
    else:
        # Fall back to synthetic if Atlanta not available
        data_fallback = generate_sparse_data(n_points=1000, noise_ratio=0.7)
        exp2_scaling(data_fallback, eps=1.0, minPts=5, k=5)

    # ── E3: Top-k quality ────────────────────────────────────────
    if "Atlanta" in real_datasets:
        exp3_topk_quality(real_datasets["Atlanta"], eps=0.5, minPts=5)
    else:
        data_varied = generate_varied_density_data(n_points=1000)
        exp3_topk_quality(data_varied, eps=1.0, minPts=5)

    # ── E4: Parameter sensitivity ────────────────────────────────
    if "Atlanta" in real_datasets:
        exp4_parameter_sensitivity(real_datasets["Atlanta"], k=3)

    # ── E5: Multi-dataset ────────────────────────────────────────
    exp5_multi_dataset(real_datasets, eps=0.5, minPts=5, k=5)

    # ── E6: Early termination ────────────────────────────────────
    # exp6_early_termination()

    # ── Visualizations ───────────────────────────────────────────
    for name, data in real_datasets.items():
        plot_clustering_comparison(
            data, eps=0.5, minPts=5, k=3,
            dataset_name=f"{name} Restaurants (OSM)",
            filename=f"viz_{name.lower()}_k3.png"
        )

    print("\n" + "=" * 60)
    print(f"All results saved to ./{RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()