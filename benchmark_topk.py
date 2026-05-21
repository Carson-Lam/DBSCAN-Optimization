import numpy as np
import matplotlib.pyplot as plt
import time


# MAXRS (LANCE WITH WEIGHTS)
def maxrs_sweepline_LP(points, rect_w, rect_h):
    """
    Sweep line O(n² log n) version.
    
    Key optimization: For each x-position, filter points ONCE,
    then sweep through y-positions checking only filtered points.
    
    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
    Returns:
        (best_x, best_y, max_sum)
    """
    if not points:
        return (0, 0, 0, [])
    
    # Check if points have weights
    has_weights = len(points[0]) == 3
    
    # Normalize points to always have weights
    if has_weights:
        normalized_points = points
    else:
        normalized_points = [(x, y, 1) for (x, y) in points]
    
    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(normalized_points)
    
    best_sum = -np.inf
    best_pos = None
    best_points = []

    # Add first element to list
    points_in_x_range = [points_by_x_val[0]]
    next_element_index = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    all_points_covered = False
    
    # Start left side of grid at x's
    # Output: X jumps [1,5, 6, 8, ...]
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h

        # (1) Add new points to list
        while next_element_index < len(points_by_x_val) and points_by_x_val[next_element_index][0] <= horizontal_ub:
            points_in_x_range.append(points_by_x_val[next_element_index])
            next_element_index += 1

        # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
        if next_element_index == len(points_by_x_val):
            all_points_covered = True

        # (2) Sort list by y-values
        # THEORY: I think this is very fast because the list is already mostly sorted
        points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

        # (3) Iterate until we reach a point in range vertically
        i = 0
        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i][1] < vertical_lb:
                i += 1

        # (4) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = current_pos
            # Collect points in best rectangle
            best_points = [
                pt for pt in points  # Use original points list
                if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
                and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
            ]
        
        # (5) Iterate through points, altering window as we go, until we use the current left-most point as bottom
        while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
            # Remove weight of previous bottom point
            current_sum -= points_in_x_range[i][2]
            # Shift i to next point and reset current pos
            i += 1
            current_pos = (p[0], points_in_x_range[i][1])
            # Shift j and add new points
            while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
                current_sum += points_in_x_range[j][2]
                j += 1
            # Check new window
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = current_pos

        # (6) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
        if not all_points_covered:
            while i < len(points_in_x_range) and points_in_x_range[i] != p:
                i += 1
            if i < len(points_in_x_range):
                del points_in_x_range[i]
        else:
            break
    
    return best_pos + (best_sum, best_points)


# DBSCAN
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
    """
    Optimized DBSCAN using MaxRS to find densest regions first.
    """
    labels = {tuple(P): None for P in DB}
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    # Counters (DEBUG)
    outer_loop_iterations = 0
    maxrs_calls = 0
    range_queries = 0

    # Calculate rectangle sizes
    rect_w2, rect_h2 = 2 * eps, 2 * eps

    # Outer Loop: Repeated find densest region w MaxRS
    while unlabeled and (max_iterations is None or C < max_iterations):     
        outer_loop_iterations += 1
        unlabeled_list = list(unlabeled)

        # M' - Larger MaxRS rect (outside DBSCAN circle)
        x2, y2, sum2, points2 = maxrs_sweepline_LP(unlabeled_list, rect_w2, rect_h2)
        print(f"Iteration {outer_loop_iterations}: MaxRS sum={sum2}, unlabeled={len(unlabeled)}")
        maxrs_calls += 1

        # CASE 1: |M'| < minpoints so terminate DBSCAN
        if sum2 < minPts:
            break
                
        # CASE 2: there may be a cluster
        P = tuple(points2[0])

        # Check if P is a core point by finding its neighbors
        N = RangeQuery(DB, distFunc, P, eps) 
        range_queries += 1

        # P does not have enough neighbors, Mark as noise (-1)
        if len(N) < minPts:
            labels[P] = -1 
            unlabeled.discard(P)
            continue

        # P has enough neighbors, ExpandCluster on P
        C += 1
        labels[P] = C
        unlabeled.discard(P)

        # Seed set - all neighbors of P except itself
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

    # If we stopped early due to max_iterations, mark remaining as noise
    if unlabeled:
        for P in unlabeled:
            labels[P] = -1

    return labels, outer_loop_iterations, maxrs_calls, range_queries


def DBSCAN(DB, distFunc, eps, minPts, max_iterations=None):
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
    
    # Mark any remaining unlabeled points as noise
    for P in DB:
        P_tuple = tuple(P)
        if labels[P_tuple] is None:
            labels[P_tuple] = -1

    return labels, main_loop_iterations, range_queries


# DATA GENERATION
def generate_data():
    np.random.seed(42)
    cluster1 = np.random.randn(20, 2) * 0.5 + [2, 2]
    cluster2 = np.random.randn(20, 2) * 0.5 + [6, 6]
    cluster3 = np.random.randn(20, 2) * 0.5 + [10, 2]
    noise = np.random.uniform(low=0, high=12, size=(10, 2))
    return np.vstack((cluster1, cluster2, cluster3, noise))


def generate_sparse_data(n_points=1000, n_clusters=3, noise_ratio=0.7):
    np.random.seed(42)
    n_cluster_points = int(n_points * (1 - noise_ratio))
    points_per_cluster = n_cluster_points // n_clusters
    
    clusters = []
    centers = [(5, 5), (15, 15), (25, 5)]
    
    for i in range(n_clusters):
        cluster = np.random.randn(points_per_cluster, 2) * 0.5 + centers[i]
        clusters.append(cluster)
    
    n_noise = n_points - (points_per_cluster * n_clusters)
    noise = np.random.uniform(low=0, high=30, size=(n_noise, 2))
    
    return np.vstack(clusters + [noise])


def generate_dense_data(n_points=1000, n_clusters=10):
    np.random.seed(42)
    points_per_cluster = n_points // n_clusters
    
    clusters = []
    for i in range(n_clusters):
        center = np.random.uniform(0, 30, 2)
        cluster = np.random.randn(points_per_cluster, 2) * 0.5 + center
        clusters.append(cluster)
    
    return np.vstack(clusters)

def generate_varied_density_data(n_points=1000):
    """
    Generate clusters with DIFFERENT sizes so density order is not creation order.
    This properly tests the density-first advantage.
    """
    np.random.seed(42)
    
    cluster1 = np.random.randn(20, 2) * 0.5 + [5, 5]
    
    cluster2 = np.random.randn(50, 2) * 0.5 + [15, 15]
    
    cluster3 = np.random.randn(100, 2) * 0.5 + [25, 5]
    
    # Another medium cluster (60 points) - created FOURTH
    cluster4 = np.random.randn(60, 2) * 0.5 + [10, 25]
    
    # Small cluster (30 points) - created FIFTH
    cluster5 = np.random.randn(30, 2) * 0.5 + [20, 10]
    
    # Noise
    noise = np.random.uniform(low=0, high=30, size=(n_points - 260, 2))
    
    return np.vstack((cluster1, cluster2, cluster3, cluster4, cluster5, noise))


# VISUALIZATION FUNCTIONS
def plot_comparison(ax, data, labels, title, max_iterations):
    """
    Plot clusters on a given axis (for side-by-side comparison).
    """
    # Get unique cluster labels (excluding noise -1 and None)
    clusters = sorted([c for c in set(labels.values()) if c not in [-1, None]])
    
    # Create color map
    colors = plt.cm.tab10(range(len(clusters)))
    
    # Plot each cluster
    for i, cluster_id in enumerate(clusters):
        cluster_points = [p for p in data if labels[tuple(p)] == cluster_id]
        if cluster_points:
            cluster_array = np.array(cluster_points)
            ax.scatter(cluster_array[:, 0], cluster_array[:, 1], 
                       c=[colors[i]], label=f'Cluster {cluster_id} ({len(cluster_points)} pts)', 
                       edgecolor='k', s=80, alpha=0.7)
    
    # Plot noise points
    noise_points = [p for p in data if labels[tuple(p)] == -1]
    if noise_points:
        noise_array = np.array(noise_points)
        ax.scatter(noise_array[:, 0], noise_array[:, 1], 
                   c='lightgray', label=f'Noise ({len(noise_points)} pts)', 
                   marker='x', s=50, alpha=0.5)
    
    ax.set_title(f"{title}\n{len(clusters)} clusters (max_iterations={max_iterations})")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)


def benchmark_and_visualize(dataset_name, data, eps, minPts, max_iterations=2, runs=5):
    """
    Run both algorithms, print statistics, and visualize the top-k clusters.
    """
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Points: {len(data)}, eps: {eps}, minPts: {minPts}")
    print(f"Max Iterations: {max_iterations}")
    print(f"{'='*60}")
    
    # Original DBSCAN
    stats_orig = []
    times_orig = []
    for _ in range(runs):
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
    for _ in range(runs):
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
    print(f"\nOriginal DBSCAN:")
    print(f"  Avg Runtime: {avg_orig:.6f}s (±{np.std(times_orig):.6f}s)")
    print(f"  Clusters: {clusters_orig}, Noise: {noise_orig}")
    print(f"  Main Loop Iterations: {avg_main_loops:.1f}")
    print(f"  Range Queries: {avg_rq_orig:.1f}")
    
    print(f"\nOptimized DBSCAN:")
    print(f"  Avg Runtime: {avg_opt:.6f}s (±{np.std(times_opt):.6f}s)")
    print(f"  Clusters: {clusters_opt}, Noise: {noise_opt}")
    print(f"  Outer Loop Iterations: {avg_outer_loops:.1f}")
    print(f"  MaxRS Calls: {avg_maxrs:.1f}")
    print(f"  Range Queries: {avg_rq_opt:.1f}")
    
    speedup = avg_orig / avg_opt
    rq_reduction = (1 - avg_rq_opt / avg_rq_orig) * 100

    if speedup > 1:
        print(f"\n Optimized is {speedup:.2f}x FASTER")
    else:
        print(f"\n Optimized is {1/speedup:.2f}x SLOWER")

    print(f"Range Query Reduction: {rq_reduction:.1f}%")
    print(f"Loop Iterations: {avg_main_loops:.0f} → {avg_outer_loops:.0f}")
    
    # Visualize the top-k clusters
    # Create side-by-side comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Plot Regular DBSCAN results
    plot_comparison(ax1, data, labels_orig, 
                    f"Regular DBSCAN (Random Order)\n{dataset_name}", 
                    max_iterations)

    # Plot Optimized DBSCAN results  
    plot_comparison(ax2, data, labels_opt, 
                    f"Optimized DBSCAN (Density-First)\n{dataset_name}", 
                    max_iterations)

    plt.tight_layout()
    plt.show()
    
    return labels_orig, labels_opt


# MAIN EXECUTION
if __name__ == '__main__':
    print("="*60)
    print("DBSCAN OPTIMIZATION: Top-K Cluster Visualization")
    print("="*60)
    
    # Test 1: Small dataset (3 actual clusters)
    print("\n>>> TEST 1: Small Dataset")
    data_small = generate_data()
    benchmark_and_visualize("Small Dataset (3 clusters)", data_small, 
                           eps=1.0, minPts=5, max_iterations=2)
    
    # Test 2: Sparse dataset (3 clusters, 70% noise)
    print("\n>>> TEST 2: Sparse Dataset")
    data_sparse = generate_sparse_data(n_points=1000, noise_ratio=0.7)
    benchmark_and_visualize("Sparse Dataset (3 clusters, 70% noise)", data_sparse, 
                           eps=1.0, minPts=5, max_iterations=2)
    
    # Test 3: Dense dataset (10 clusters) 
    print("\n>>> TEST 3: Dense Dataset")
    data_varied = generate_varied_density_data(n_points=1000)
    benchmark_and_visualize("Varied Density Dataset (5 clusters, different sizes)", data_varied, 
                        eps=1.0, minPts=5, max_iterations=2)
        
    # Test 4: Large sparse dataset
    print("\n>>> TEST 4: Large Sparse Dataset")
    data_large = generate_sparse_data(n_points=1500, noise_ratio=0.8)
    benchmark_and_visualize("Large Sparse Dataset (1500 pts, 80% noise)", data_large, 
                           eps=1.0, minPts=5, max_iterations=2)
    
    print("\n" + "="*60)
    print("All tests complete!")
    print("="*60)