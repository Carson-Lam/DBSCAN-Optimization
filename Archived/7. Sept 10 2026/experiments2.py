"""
experiments2.py
===============
Synthetic dataset experiments for the paper.
Generates a configurable synthetic point set and runs:
  - E2: Scaling (runtime + range queries vs. n)
  - Visualization map (side-by-side cluster comparison)

All outputs saved to ./results2/

Parameters (edit in CONFIG below):
  ALPHA       -- noise fraction (default 0.90 → 90% noise, 10% cluster pts)
  N_CLUSTERS  -- number of clusters in the 1-alpha cluster points
  STDEV       -- standard deviation of each Gaussian cluster
  N_FULL      -- total number of points for the full dataset
  K           -- number of top-k clusters to find
  EPS         -- DBSCAN neighborhood radius (in [0,1] space)
  MIN_PTS     -- DBSCAN minimum points threshold

Dependencies: numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import csv

from benchmark_topk import (
    DBSCAN,
    DBSCAN_Optimized,
    euclidean_distance,
)

# ── Output directory ──────────────────────────────────────────────────────────
RESULTS_DIR = "./results2"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG 
# ═══════════════════════════════════════════════════════════════════════════════
N_FULL     = 10000  # num of dataset points  
ALPHA      = 0.90   # % of noise pts
N_CLUSTERS = 5      # num of clusters 
STDEV      = 0.005  # Tightness of clusters, 0.005 = very tight clusters
EPS        = 0.010  # EPS range for MaxRS and DBSCAN 2× STDEV 
MIN_PTS    = 5      # MinPTS for DBSCAN 
K          = 3      # num of top-k clusters 
RUNS       = 3      # num of runs to avg for timing and range query counts

# ═══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

def generate_synthetic(n_total, alpha, n_clusters, stdev, seed=42):
    """
    Generate a synthetic 2D point set in [0, 1] x [0, 1].

    Parameters
    ----------
    n_total    : total number of points
    alpha      : noise fraction (e.g. 0.90 → 90% noise)
    n_clusters : number of Gaussian clusters
    stdev      : standard deviation of each cluster
    seed       : random seed for reproducibility

    Returns
    -------
    data       : np.ndarray of shape (n_total, 2), all points in [0,1]^2
    centers    : list of (x, y) cluster centroids
    """
    rng = np.random.default_rng(seed)

    n_noise        = int(n_total * alpha)
    n_cluster_pts  = n_total - n_noise
    pts_per_cluster = n_cluster_pts // n_clusters
    remainder       = n_cluster_pts - pts_per_cluster * n_clusters

    # Cluster centroids placed uniformly at random in [0.1, 0.9]
    # (margin avoids clusters right at the border)
    centers = rng.uniform(0.1, 0.9, size=(n_clusters, 2))

    cluster_arrays = []
    for i, center in enumerate(centers):
        # Last cluster gets any leftover points from integer division
        n_pts = pts_per_cluster + (remainder if i == n_clusters - 1 else 0)
        pts = rng.normal(loc=center, scale=stdev, size=(n_pts, 2))
        # Clip to [0, 1] to keep all points in the unit square
        pts = np.clip(pts, 0.0, 1.0)
        cluster_arrays.append(pts)

    # Noise: uniform over [0, 1]^2
    noise = rng.uniform(0.0, 1.0, size=(n_noise, 2))

    data = np.vstack(cluster_arrays + [noise])

    # Shuffle so cluster points and noise are not ordered
    idx = rng.permutation(len(data))
    data = data[idx]

    return data, centers.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def run_vanilla_full(data, eps, minPts, runs=RUNS):
    """Vanilla DBSCAN to full completion, no k cap."""
    times, rqs = [], []
    for _ in range(runs):
        t0 = time.time()
        labels, _, rq = DBSCAN(data, euclidean_distance, eps, minPts,
                               max_iterations=None)
        times.append(time.time() - t0)
        rqs.append(rq)
    return {
        "time_mean": np.mean(times),
        "time_std":  np.std(times),
        "rq_mean":   np.mean(rqs),
        "labels":    labels,
    }


def run_vanilla_k(data, eps, minPts, k, runs=RUNS):
    """Vanilla DBSCAN stopped at k clusters."""
    times, rqs = [], []
    for _ in range(runs):
        t0 = time.time()
        labels, _, rq = DBSCAN(data, euclidean_distance, eps, minPts,
                               max_iterations=k)
        times.append(time.time() - t0)
        rqs.append(rq)
    return {
        "time_mean": np.mean(times),
        "time_std":  np.std(times),
        "rq_mean":   np.mean(rqs),
        "labels":    labels,
    }


def run_gdbscan_k(data, eps, minPts, k, runs=RUNS):
    """GDBSCAN stopped at k clusters (density-first)."""
    times, rqs = [], []
    for _ in range(runs):
        t0 = time.time()
        labels, _, _, rq = DBSCAN_Optimized(data, euclidean_distance, eps, minPts,
                                            max_iterations=k)
        times.append(time.time() - t0)
        rqs.append(rq)
    return {
        "time_mean": np.mean(times),
        "time_std":  np.std(times),
        "rq_mean":   np.mean(rqs),
        "labels":    labels,
    }


def cluster_sizes(labels):
    ids = set(labels.values()) - {-1, None}
    return sorted([sum(1 for v in labels.values() if v == i) for i in ids], reverse=True)


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
# EXPERIMENT: SCALING
# Subsample full dataset at 10/25/50/75/100% and measure all three algorithms.
# Three lines: vanilla full, vanilla-k, GDBSCAN-k
# ═══════════════════════════════════════════════════════════════════════════════

def exp_scaling(full_data, eps, minPts, k):
    print(f"\n[Scaling] n_full={len(full_data)}, eps={eps}, minPts={minPts}, k={k}")
    fractions = [0.10, 0.25, 0.50, 0.75, 1.00]
    n_full = len(full_data)

    ns = []
    stats_full, stats_vk, stats_ok = [], [], []
    rows = []

    rng = np.random.default_rng(0)

    for frac in fractions:
        n = max(50, int(n_full * frac))
        idx = rng.choice(n_full, n, replace=False)
        subset = full_data[idx]

        print(f"  n={n}...")
        sf = run_vanilla_full(subset, eps, minPts)
        sv = run_vanilla_k(subset, eps, minPts, k)
        so = run_gdbscan_k(subset, eps, minPts, k)

        ns.append(n)
        stats_full.append(sf)
        stats_vk.append(sv)
        stats_ok.append(so)

        rows.append([
            n, frac,
            sf["rq_mean"],  sf["time_mean"],
            sv["rq_mean"],  sv["time_mean"],
            so["rq_mean"],  so["time_mean"],
        ])
        print(f"    vanilla_full  rq={sf['rq_mean']:.0f}  t={sf['time_mean']:.3f}s")
        print(f"    vanilla_k     rq={sv['rq_mean']:.0f}  t={sv['time_mean']:.3f}s")
        print(f"    gdbscan_k     rq={so['rq_mean']:.0f}  t={so['time_mean']:.3f}s")

    save_csv(rows,
             ["n", "fraction",
              "vanilla_full_rq", "vanilla_full_time",
              "vanilla_k_rq",    "vanilla_k_time",
              "gdbscan_k_rq",    "gdbscan_k_time"],
             "e2_scaling_synthetic.csv")

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Runtime
    ax1.plot(ns, [s["time_mean"] for s in stats_full], "o-",
             label="Vanilla DBSCAN (full, no k)", color="#2196F3",
             linewidth=2, markersize=8)
    ax1.plot(ns, [s["time_mean"] for s in stats_vk], "s--",
             label=f"Vanilla DBSCAN (top-k, k={k})", color="#9C27B0",
             linewidth=2, markersize=8)
    ax1.plot(ns, [s["time_mean"] for s in stats_ok], "^-",
             label=f"GDBSCAN (top-k, k={k})", color="#FF5722",
             linewidth=2, markersize=8)
    ax1.set_xlabel("Number of points (n)", fontsize=12)
    ax1.set_ylabel("Runtime (seconds)", fontsize=12)
    ax1.set_title("Runtime vs. Dataset Size", fontsize=13)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Range queries
    ax2.plot(ns, [s["rq_mean"] for s in stats_full], "o-",
             label="Vanilla DBSCAN (full, no k)", color="#2196F3",
             linewidth=2, markersize=8)
    ax2.plot(ns, [s["rq_mean"] for s in stats_vk], "s--",
             label=f"Vanilla DBSCAN (top-k, k={k})", color="#9C27B0",
             linewidth=2, markersize=8)
    ax2.plot(ns, [s["rq_mean"] for s in stats_ok], "^-",
             label=f"GDBSCAN (top-k, k={k})", color="#FF5722",
             linewidth=2, markersize=8)
    ax2.set_xlabel("Number of points (n)", fontsize=12)
    ax2.set_ylabel("Range queries issued", fontsize=12)
    ax2.set_title("Range Queries vs. Dataset Size", fontsize=13)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"Scaling — Synthetic (α={ALPHA}, clusters={N_CLUSTERS}, "
        f"σ={STDEV}, ε={eps}, minPts={minPts}, k={k})",
        fontsize=12)
    plt.tight_layout()
    save_fig(fig, "e2_scaling_synthetic.png")

    return stats_full[-1], stats_vk[-1], stats_ok[-1]  # Return full-n stats


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION MAP
# Side-by-side: vanilla-k vs GDBSCAN-k on the full dataset.
# Also marks true cluster centroids for reference.
# ═══════════════════════════════════════════════════════════════════════════════

def exp_visualization(full_data, centers, eps, minPts, k):
    print(f"\n[Visualization] Running on full dataset (n={len(full_data)})...")

    sv = run_vanilla_k(full_data, eps, minPts, k, runs=1)
    so = run_gdbscan_k(full_data, eps, minPts, k, runs=1)

    lv = sv["labels"]
    lo = so["labels"]

    print(f"  Vanilla-k:  {sv['n_clusters'] if 'n_clusters' in sv else len(set(lv.values()) - {-1, None})} clusters")
    print(f"  GDBSCAN-k:  {len(set(lo.values()) - {-1, None})} clusters")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    def _plot(ax, labels, title):
        cids = sorted(set(labels.values()) - {-1, None})
        colors = plt.cm.tab10(range(len(cids)))

        for i, cid in enumerate(cids):
            pts = np.array([p for p in full_data if labels[tuple(p)] == cid])
            ax.scatter(pts[:, 0], pts[:, 1],
                       c=[colors[i]],
                       label=f"Cluster {i+1} ({len(pts)} pts)",
                       edgecolor="k", linewidths=0.3,
                       s=15, alpha=0.7, zorder=3)

        noise = np.array([p for p in full_data if labels[tuple(p)] == -1])
        if len(noise):
            ax.scatter(noise[:, 0], noise[:, 1],
                       c="lightgray", marker=".", s=5,
                       alpha=0.3, label=f"Noise ({len(noise)} pts)", zorder=1)

        # Mark true centroids
        for j, (cx, cy) in enumerate(centers):
            ax.scatter(cx, cy, marker="*", s=200, c="black",
                       zorder=5, linewidths=0.5)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.2)

    n_v_clusters = len(set(lv.values()) - {-1, None})
    n_o_clusters = len(set(lo.values()) - {-1, None})

    _plot(ax1, lv,
          f"Vanilla DBSCAN (stopped at k={k})\n"
          f"{n_v_clusters} clusters found  |  "
          f"sizes: {cluster_sizes(lv)}")

    _plot(ax2, lo,
          f"GDBSCAN (density-first, k={k})\n"
          f"{n_o_clusters} clusters found  |  "
          f"sizes: {cluster_sizes(lo)}")

    fig.suptitle(
        f"Synthetic Dataset — α={ALPHA}, {N_CLUSTERS} clusters, "
        f"σ={STDEV}, n={len(full_data)}\n"
        f"ε={eps}, minPts={minPts}  |  ★ = true cluster centroid",
        fontsize=12)
    plt.tight_layout()
    save_fig(fig, "viz_synthetic.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("experiments2.py — Synthetic Dataset Experiments")
    print("=" * 60)
    print(f"\nConfig:")
    print(f"  Alpha (noise):  {ALPHA:.0%}")
    print(f"  Clusters:       {N_CLUSTERS}")
    print(f"  Stdev:          {STDEV}")
    print(f"  N (full):       {N_FULL}")
    print(f"  k:              {K}")
    print(f"  eps:            {EPS}")
    print(f"  minPts:         {MIN_PTS}")

    # Generate dataset
    print(f"\nGenerating synthetic dataset ({N_FULL} points)...")
    data, centers = generate_synthetic(N_FULL, ALPHA, N_CLUSTERS, STDEV)
    print(f"  {int(N_FULL * (1 - ALPHA))} cluster points across {N_CLUSTERS} clusters")
    print(f"  {int(N_FULL * ALPHA)} noise points")
    print(f"  Centroids: {[f'({c[0]:.2f}, {c[1]:.2f})' for c in centers]}")

    # Scaling experiment
    exp_scaling(data, EPS, MIN_PTS, K)

    # Visualization
    exp_visualization(data, centers, EPS, MIN_PTS, K)

    print(f"\n{'=' * 60}")
    print(f"Done. Results saved to ./{RESULTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()