"""
Run DBSCAN clustering on real OpenStreetMap restaurant data.

This file loads the restaurant data fetched by fetch_osm_data.py
and runs both regular and optimized (MaxRS) DBSCAN algorithms.

Usage:
    python cluster_osm_data.py

"""

import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os

# Get Restaurant data from osm API JSON
def load_osm_data(filename='atlanta_restaurants_osm.json'):
    """ Get Restaurant data from osm API JSON """
    if not os.path.exists(filename):
        print(f" File not found: {filename}")
        print(f" Run fetch_osm_data.py first to download the data.")
        return None
    
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f" Loaded {len(data)} restaurants from {filename}")
    return data

# Convert JSON lat and long to x and y for points
def convert_to_xy(restaurants, bbox):
    """
    Convert (lon, lat) to approximate (x, y) in km for clustering using equirectangular projection
    
    Args:
        restaurants: list of dicts with 'lon' and 'lat' keys
        bbox: (min_lat, min_lon, max_lat, max_lon)
    
    Returns:
        numpy array of (x, y) in km
    """
    min_lat, min_lon, max_lat, max_lon = bbox # Atlanta box

    center_lat = (min_lat + max_lat) / 2 # Find center of box
    
    km_per_deg_lat = 111.0  # Constant based on Earth's circumference
    km_per_deg_lon = 111.0 * np.cos(np.radians(center_lat)) # Constant based on Earth's circumference + shrinkage
    
    points = []
    for r in restaurants:
        # Get restaurant lat long values from JSON
        lon = r['lon'] 
        lat = r['lat']
        x = (lon - min_lon) * km_per_deg_lon # Subtract leftmost longitude (x) of box, mult by constant 
        y = (lat - min_lat) * km_per_deg_lat # Subtract bottommost latitude (y) of box, mult by constant
        points.append([x, y])
    
    return np.array(points)


# MAXRS (LANCE WITH WEIGHTS)
def maxrs_sweepline_LP(points, rect_w, rect_h):
    """
    Sweep line O(n² log n) version for finding maximum range sum.
    """
    if not points:
        return (0, 0, 0, [])
    
    has_weights = len(points[0]) == 3
    
    if has_weights:
        normalized_points = points
    else:
        normalized_points = [(x, y, 1) for (x, y) in points]
    
    points_by_x_val = sorted(normalized_points)
    
    best_sum = -np.inf
    best_pos = None
    best_points = []

    points_in_x_range = [points_by_x_val[0]]
    next_element_index = 1
    all_points_covered = False
    
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h

        while next_element_index < len(points_by_x_val) and points_by_x_val[next_element_index][0] <= horizontal_ub:
            points_in_x_range.append(points_by_x_val[next_element_index])
            next_element_index += 1

        if next_element_index == len(points_by_x_val):
            all_points_covered = True

        points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

        i = 0
        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i][1] < vertical_lb:
                i += 1

        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = current_pos
            best_points = [
                pt for pt in points
                if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
                and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
            ]
        
        while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
            current_sum -= points_in_x_range[i][2]
            i += 1
            current_pos = (p[0], points_in_x_range[i][1])
            while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
                current_sum += points_in_x_range[j][2]
                j += 1
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = current_pos

        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i] != p:
                i += 1
            if i < len(points_in_x_range):
                del points_in_x_range[i]
        else:
            break
    
    return best_pos + (best_sum, best_points)


# DBSCAN FUNCTIONS
def RangeQuery(DB, distFunc, Q, eps):
    N = []
    Q_tuple = tuple(Q)
    for P in DB:
        P_tuple = tuple(P)
        if distFunc(Q_tuple, P_tuple) <= eps:
            N.append(P_tuple)
    return N


def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def DBSCAN_Optimized(DB, distFunc, eps, minPts, max_iterations=None):
    """Optimized DBSCAN using MaxRS to find densest regions first."""
    labels = {tuple(P): None for P in DB}
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    outer_loop_iterations = 0
    maxrs_calls = 0
    range_queries = 0

    rect_w2, rect_h2 = 2 * eps, 2 * eps

    while unlabeled and (max_iterations is None or C < max_iterations):     
        outer_loop_iterations += 1
        unlabeled_list = list(unlabeled)

        x2, y2, sum2, points2 = maxrs_sweepline_LP(unlabeled_list, rect_w2, rect_h2)
        maxrs_calls += 1

        if sum2 < minPts:
            break
                
        P = tuple(points2[0])

        N = RangeQuery(DB, distFunc, P, eps) 
        range_queries += 1

        if len(N) < minPts:
            labels[P] = -1 
            unlabeled.discard(P)
            continue

        C += 1
        labels[P] = C
        unlabeled.discard(P)

        S = set(N) - {P}

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C  
            if labels[Q] is not None: continue
            labels[Q] = C
            unlabeled.discard(Q)
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    if unlabeled:
        for P in unlabeled:
            labels[P] = -1

    return labels, outer_loop_iterations, maxrs_calls, range_queries


def DBSCAN(DB, distFunc, eps, minPts, max_iterations=None):
    """Regular DBSCAN algorithm."""
    labels = {tuple(P): None for P in DB}
    C = 0
    main_loop_iterations = 0
    range_queries = 0
    clusters_found = 0

    for P in DB:
        main_loop_iterations += 1
        P_tuple = tuple(P)
        if labels[P_tuple] is not None:
            continue 

        N = RangeQuery(DB, distFunc, P, eps) 
        range_queries += 1

        if len(N) < minPts:
            labels[P_tuple] = -1
            continue

        if max_iterations is not None and clusters_found >= max_iterations:
            break 

        C += 1
        clusters_found += 1
        labels[P_tuple] = C  
        S = set(tuple(N)) - {P_tuple}

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C
            if labels[Q] is not None: continue
            labels[Q] = C
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)
    
    for P in DB:
        P_tuple = tuple(P)
        if labels[P_tuple] is None:
            labels[P_tuple] = -1

    return labels, main_loop_iterations, range_queries


def plot_comparison_geo(ax, data, labels, title, max_iterations):
    clusters = sorted([c for c in set(labels.values()) if c not in [-1, None]])
    
    colors = plt.cm.tab10(range(len(clusters)))
    
    for i, cluster_id in enumerate(clusters):
        cluster_points = [p for p in data if labels[tuple(p)] == cluster_id]
        if cluster_points:
            cluster_array = np.array(cluster_points)
            ax.scatter(cluster_array[:, 0], cluster_array[:, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id} ({len(cluster_points)} pts)', 
                       edgecolor='k', s=80, alpha=0.7)
    
    noise_points = [p for p in data if labels[tuple(p)] == -1]
    if noise_points:
        noise_array = np.array(noise_points)
        ax.scatter(noise_array[:, 0], noise_array[:, 1], 
                   c='lightgray', label=f'Noise ({len(noise_points)} pts)', 
                   marker='x', s=50, alpha=0.5)
    
    ax.set_title(f"{title}\n{len(clusters)} clusters (max_iterations={max_iterations})")
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)


def benchmark_and_visualize(data, eps, minPts, max_iterations, runs=3):
    """Run both algorithms and visualize results."""
    print(f"\n{'='*60}")
    print(f"Restaurant Clustering - Atlanta, GA (Real OSM Data)")
    print(f"Points: {len(data)}, eps: {eps}km, minPts: {minPts}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*60}")
    
    # Original DBSCAN
    stats_orig = []
    times_orig = []
    for run in range(runs):
        print(f"  Running Regular DBSCAN... ({run+1}/{runs})")
        start = time.time()
        labels_orig, main_loops, rq_orig = DBSCAN(data, euclidean_distance, eps, minPts, max_iterations=max_iterations)
        times_orig.append(time.time() - start)
        stats_orig.append((main_loops, rq_orig))

    avg_orig = np.mean(times_orig)
    clusters_orig = len(set(labels_orig.values()) - {-1, None})
    noise_orig = sum(1 for v in labels_orig.values() if v == -1)
    avg_main_loops = np.mean([s[0] for s in stats_orig])
    avg_rq_orig = np.mean([s[1] for s in stats_orig])
    
    # Optimized DBSCAN
    times_opt = []
    stats_opt = []
    for run in range(runs):
        print(f"  Running Optimized DBSCAN... ({run+1}/{runs})")
        start = time.time()
        labels_opt, outer_loops, maxrs, rq_opt = DBSCAN_Optimized(
            data, euclidean_distance, eps, minPts, max_iterations=max_iterations
        )
        times_opt.append(time.time() - start)
        stats_opt.append((outer_loops, maxrs, rq_opt))
    
    avg_opt = np.mean(times_opt)
    clusters_opt = len(set(labels_opt.values()) - {-1, None})
    noise_opt = sum(1 for v in labels_opt.values() if v == -1)
    avg_outer_loops = np.mean([s[0] for s in stats_opt])
    avg_maxrs = np.mean([s[1] for s in stats_opt])
    avg_rq_opt = np.mean([s[2] for s in stats_opt])
    
    # Print Results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    
    print(f"\nOriginal DBSCAN:")
    print(f"  Avg Runtime: {avg_orig:.4f}s (±{np.std(times_orig):.4f}s)")
    print(f"  Clusters Found: {clusters_orig}")
    print(f"  Noise Points: {noise_orig}")
    print(f"  Main Loop Iterations: {avg_main_loops:.0f}")
    print(f"  Range Queries: {avg_rq_orig:.0f}")
    
    print(f"\nOptimized DBSCAN:")
    print(f"  Avg Runtime: {avg_opt:.4f}s (±{np.std(times_opt):.4f}s)")
    print(f"  Clusters Found: {clusters_opt}")
    print(f"  Noise Points: {noise_opt}")
    print(f"  Outer Loop Iterations: {avg_outer_loops:.0f}")
    print(f"  MaxRS Calls: {avg_maxrs:.0f}")
    print(f"  Range Queries: {avg_rq_opt:.0f}")
    
    speedup = avg_orig / avg_opt
    rq_reduction = (1 - avg_rq_opt / avg_rq_orig) * 100

    print(f"\n{'='*60}")
    if speedup > 1:
        print(f" Optimized is {speedup:.2f}x FASTER")
    else:
        print(f" Optimized is {1/speedup:.2f}x SLOWER")

    print(f"Range Query Reduction: {rq_reduction:.1f}%")
    print(f"Loop Iterations: {avg_main_loops:.0f} → {avg_outer_loops:.0f}")
    print(f"{'='*60}")
    
    # Get cluster sizes for comparison
    print(f"\nCluster Size Comparison:")
    print(f"{'Cluster':<10} {'Regular':<15} {'Optimized':<15}")
    print("=" * 40)
    
    for i in range(1, max(clusters_orig, clusters_opt) + 1):
        orig_size = sum(1 for v in labels_orig.values() if v == i)
        opt_size = sum(1 for v in labels_opt.values() if v == i)
        if orig_size > 0 or opt_size > 0:
            print(f"Cluster {i:<3} {orig_size:<15} {opt_size:<15}")
    
    # Visualize
    print(f"\nGenerating visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    plot_comparison_geo(ax1, data, labels_orig, 
                        f"Regular DBSCAN (Random Order)\nAtlanta Restaurants (OSM)", 
                        max_iterations)

    plot_comparison_geo(ax2, data, labels_opt, 
                        f"Optimized DBSCAN (Density-First)\nAtlanta Restaurants (OSM)", 
                        max_iterations)

    plt.tight_layout()

    # Save output to png file
    output_file = 'atlanta_osm_clustering_results.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f" Visualization saved to {output_file}")
    plt.show()
    
    return labels_orig, labels_opt


def main():
    print("="*60)
    print("DBSCAN CLUSTERING ON REAL OSM DATA")
    print("="*60)
    
    # Load OSM data
    restaurants = load_osm_data('atlanta_restaurants_osm.json')
    
    if restaurants is None:
        return
    
    # Atlanta bounding box
    atlanta_bbox = (
        33.6490,  # min_lat (south) - Hartsfield-Jackson Airport
        -84.5510,  # min_lon (west) - Six Flags
        33.8860,  # max_lat (north) - Roswell
        -84.2890   # max_lon (east) - Stone Mountain
    )
    
    # Convert to Cartesian coordinates
    print("\nConverting to Cartesian coordinates...")
    data = convert_to_xy(restaurants, atlanta_bbox)
    print(f"Data shape: {data.shape}")
    print(f"X range: {data[:, 0].min():.2f} to {data[:, 0].max():.2f} km")
    print(f"Y range: {data[:, 1].min():.2f} to {data[:, 1].max():.2f} km")
    
    # Clustering parameters
    eps = 0.5  # 500m radius (city block)
    minPts = 5  # Minimum 5 restaurants for a cluster
    max_iterations = 3  # Find top 3 densest clusters
    
    print(f"\nClustering Parameters:")
    print(f"  eps (radius): {eps} km ({eps*1000:.0f} meters)")
    print(f"  minPts: {minPts}")
    print(f"  max_iterations: {max_iterations}")
    
    benchmark_and_visualize(data, eps, minPts, max_iterations, runs=3)
    
    print("\n" + "="*60)
    print("Analysis complete!")
    print("="*60)


if __name__ == '__main__':
    chicago_bbox = (41.6445, -87.9401, 42.0230, -87.5240)
    restaurants = load_osm_data('chicago_restaurants_osm.json')
    data = convert_to_xy(restaurants, chicago_bbox)
    
    eps = 0.5
    minPts = 5
    max_iterations = 3
    
    print(f"Loaded {len(data)} points")
    
    labels_opt, outer_loops, maxrs, rq_opt = DBSCAN_Optimized(
        data, euclidean_distance, eps, minPts, max_iterations=max_iterations
    )
    
    print(f"Outer loops completed: {outer_loops}")
    print(f"Clusters found: {len(set(labels_opt.values()) - {-1, None})}")