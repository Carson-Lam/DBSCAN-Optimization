import numpy as np
import matplotlib.pyplot as plt
import time


# MAXRS (LANCE WITH WEIGHTS)
def maxrs_sweepline_LP(points, rect_w, rect_h):
    # Let num points := n
    # Sort time = n*log(n)
    # For each candidate: n
    #   Generate points in x range: m where m <<< n (probably)
    #   Sort points in x range : m*log(m)
    #   Run sliding window: m
    # Runtime = n*log(n) + n*(m + m*log(m) + m) = n*log(n) + 2*n*m + n*m*log(m) = O(n*m*log(m))
    # Assuming small m, this could be very fast
    # THEORY: I think this is very fast because the list is already mostly sorted after the first iteration, but unclear. Maybe sorting is closer to O(m) time then
    # According to ChatGPT, sorted uses timsort which is close to linear on already sorted data. In this case, time complexity should be O(n*m) !!!
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
        # THEORY: I think this is very fast outside the first run because the list is already mostly sorted
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
def RangeQuery(DB, distFunc, Q, eps): #Where epsilon ε defines the minimum size of a cluster
    N = []
    Q_tuple = tuple(Q) # Convert Q to tuple for Hashability
    for P in DB:
        P_tuple = tuple(P) # Convert P to tuple for Hashability
        if distFunc(Q_tuple, P_tuple) <= eps:
            N.append(P_tuple)
        
    return N

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def DBSCAN_Optimized(DB, distFunc, eps, minPts, max_iterations=None):
    """
    Optimized DBSCAN using MaxRS to find densest regions first.
    Uses Option 1: Select any point inside M' as seed.
    """
    # labels = {P: None for P in DB} #Tracks labels for each point ---  GENERIC
    labels = {tuple(P): None for P in DB} # Convert P to tuple for hashing 
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    # Counters (DEBUG)
    outer_loop_iterations = 0
    maxrs_calls = 0
    range_queries = 0

    # Calculate rectangle sizes
    rect_w1, rect_h1 = eps * np.sqrt(2), eps * np.sqrt(2) 
    rect_w2, rect_h2 = 2 * eps, 2 * eps

    # Outer Loop: Repeated find densest region w MaxRS
    while unlabeled and (max_iterations is None or outer_loop_iterations < max_iterations):     

        outer_loop_iterations += 1
        unlabedList = list(unlabeled)

        # M - Smaller MaxRS rect (inside DBSCAN circle) 
        # x1, y1, sum1, points1 = maxrs_sweepline_LP(unlabedList, rect_w1, rect_h1)
        # maxrs_calls += 1

        # M' - Larger MaxRS rect (outside DBSCAN circle)
        x2, y2, sum2, points2 = maxrs_sweepline_LP(unlabedList, rect_w2, rect_h2)
        maxrs_calls += 1

        # CASE 1: |M'| < minpoints so terminate DBSCAN, return clusters, everything else noise
        if sum2 < minPts:
            break
                
        # CASE 2: there may be a cluster
        # Option 1: Take any point inside M' as seed point P
        P = tuple(points2[0])

        # Check if P is a core point by finding it's neighbors
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

def DBSCAN(DB, distFunc, eps, minPts):
    # labels = {P: None for P in DB} #Tracks labels for each point ---  GENERIC
    labels = {tuple(P): None for P in DB} # Convert P to tuple for hashing 
    C = 0

    # Counters for analysis
    main_loop_iterations = 0
    range_queries = 0

    for P in DB:
        main_loop_iterations += 1
        P_tuple = tuple(P) # Convert P to tuple for hashing 
        if labels[P_tuple] is not None: #If point labeled, continue
            continue 

        #Obtain amount of points in euclidian distance (ε radius)
        N = RangeQuery(DB, distFunc, P, eps) 
        range_queries += 1

        if len(N) < minPts:
            labels[P_tuple] = -1 #Mark as noise (-1)
            continue

        C += 1 #Move onto new cluster for labeling (increments label by 1)
        labels[P_tuple] = C  #Label initial cluster point  
        S = set(tuple(N)) - {P_tuple} #Set a new set for the cluster excluding initial point

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C #Change from noise to border
            if labels[Q] is not None: continue #if already labeled, exit
            labels[Q] = C #Label neighbor as a set number C
            N = RangeQuery(DB, distFunc, Q, eps)
            range_queries += 1
            if len(N) >= minPts:
                S.update(N)

    return labels, main_loop_iterations, range_queries

def generate_data():
    # Generate synthetic clusters
    np.random.seed(42)

    # Cluster 1 (centered at (2, 2))
    cluster1 = np.random.randn(20, 2) * 0.5 + [2, 2]

    # Cluster 2 (centered at (6, 6))
    cluster2 = np.random.randn(20, 2) * 0.5 + [6, 6]

    # Cluster 3 (centered at (10, 2))
    cluster3 = np.random.randn(20, 2) * 0.5 + [10, 2]

    # Some random noise points
    noise = np.random.uniform(low=0, high=12, size=(10, 2))

    # Combine all points
    return np.vstack((cluster1, cluster2, cluster3, noise))

def generate_sparse_data(n_points=1000, n_clusters=3, noise_ratio=0.7):
    """
    Generate data with many noise points 
    """
    np.random.seed(42)
    
    n_cluster_points = int(n_points * (1 - noise_ratio))
    points_per_cluster = n_cluster_points // n_clusters
    
    clusters = []
    centers = [(5, 5), (15, 15), (25, 5)]
    
    for i in range(n_clusters):
        cluster = np.random.randn(points_per_cluster, 2) * 0.5 + centers[i]
        clusters.append(cluster)
    
    # Lots of noise
    n_noise = n_points - (points_per_cluster * n_clusters)
    noise = np.random.uniform(low=0, high=30, size=(n_noise, 2))
    
    return np.vstack(clusters + [noise])

def generate_dense_data(n_points=1000, n_clusters=10):
    """
    Generate data with most points in clusters 
    """
    np.random.seed(42)
    
    points_per_cluster = n_points // n_clusters
    
    clusters = []
    for i in range(n_clusters):
        center = np.random.uniform(0, 30, 2)
        cluster = np.random.randn(points_per_cluster, 2) * 0.5 + center
        clusters.append(cluster)
    
    return np.vstack(clusters)

def benchmark(dataset_name, data, eps, minPts, runs=5):
    """Run both algorithms and compare"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"Points: {len(data)}, eps: {eps}, minPts: {minPts}")
    print(f"{'='*60}")
    
    # Original DBSCAN
    stats_orig = []
    times_orig = []
    for _ in range(runs):
        start = time.time()
        labels_orig, main_loops, rq_orig = DBSCAN(data, euclidean_distance, eps, minPts)
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
        labels_opt, outer_loops, maxrs, rq_opt = DBSCAN_Optimized(data, euclidean_distance, eps, minPts, max_iterations=5)
        times_opt.append(time.time() - start)
        stats_opt.append((outer_loops, maxrs, rq_opt))
    
    avg_opt = np.mean(times_opt)
    clusters_opt = len(set(labels_opt.values()) - {-1, None})
    noise_opt = sum(1 for v in labels_opt.values() if v == -1)

    avg_outer_loops = np.mean([s[0] for s in stats_opt])
    avg_maxrs = np.mean([s[1] for s in stats_opt])
    avg_rq_opt = np.mean([s[2] for s in stats_opt])

    # Optimized GRID DBSCAN
    # times_opt_grid = []
    # stats_opt_grid = []
    # for _ in range(runs):
    #     start = time.time()
    #     labels_opt_grid, outer_loops_g, grid_calls, rq_opt_g = DBSCAN_Optimized_Grid(data, euclidean_distance, eps, minPts)
    #     times_opt_grid.append(time.time() - start)
    #     stats_opt_grid.append((outer_loops_g, grid_calls, rq_opt_g))

    # avg_opt_grid = np.mean(times_opt_grid)
    # clusters_opt_grid = len(set(labels_opt_grid.values()) - {-1, None})
    # noise_opt_grid = sum(1 for v in labels_opt_grid.values() if v == -1)

    # avg_outer_loops_grid = np.mean([s[0] for s in stats_opt_grid])
    # avg_grid_calls = np.mean([s[1] for s in stats_opt_grid])
    # avg_rq_opt_grid = np.mean([s[2] for s in stats_opt_grid])
    
    # Results
    print(f"\nOriginal DBSCAN:")
    print(f"  Avg Runtime: {avg_orig:.6f}s (+-{np.std(times_orig):.6f}s)")
    print(f"  Clusters: {clusters_orig}, Noise: {noise_orig}")
    print(f"  Main Loop Iterations: {avg_main_loops:.1f}")
    print(f"  Range Queries: {avg_rq_orig:.1f}")
    
    print(f"\nOptimized DBSCAN:")
    print(f"  Avg Runtime: {avg_opt:.6f}s (+-{np.std(times_opt):.6f}s)")
    print(f"  Clusters: {clusters_opt}, Noise: {noise_opt}")
    print(f"  Outer Loop Iterations: {avg_outer_loops:.1f}")
    print(f"  MaxRS Calls: {avg_maxrs:.1f}")
    print(f"  Range Queries: {avg_rq_opt:.1f}")

    # print(f"\nOptimized DBSCAN (GRID):")
    # print(f"  Avg Runtime: {avg_opt_grid:.6f}s (+-{np.std(times_opt_grid):.6f}s)")
    # print(f"  Clusters: {clusters_opt_grid}, Noise: {noise_opt_grid}")
    # print(f"  Outer Loop Iterations: {avg_outer_loops_grid:.1f}")
    # print(f"  Grid Calls: {avg_grid_calls:.1f}")
    # print(f"  Range Queries: {avg_rq_opt_grid:.1f}")
    
    speedup = avg_orig / avg_opt
    # speedup_grid = avg_orig / avg_opt_grid
    rq_reduction = (1 - avg_rq_opt / avg_rq_orig) * 100
    # rq_reduction_grid = (1 - avg_rq_opt_grid / avg_rq_orig) * 100

    if speedup > 1:
        print(f"\nOptimized is {speedup:.2f}x FASTER")
    else:
        print(f"\nOptimized is {1/speedup:.2f}x SLOWER")

    # if speedup_grid > 1:
    #     print(f"\nGrid is {speedup_grid:.2f}x FASTER ")
    # else:
        # print(f"\nGrid is {1/speedup_grid:.2f}x SLOWER")

    print(f"\nRange Query Reduction: {rq_reduction:.1f}%")
    print(f"Loop Iterations: {avg_main_loops:.0f} --> {avg_outer_loops:.0f}")

    # print(f"\nRange Query Reduction (Grid): {rq_reduction_grid:.1f}%")
    # print(f"Loop Iterations: {avg_main_loops:.0f} --> {avg_outer_loops_grid:.0f}")
    
    return labels_orig, labels_opt # labels_opt_grid

def plot_raw_data(data, title):
    """Plot raw data points without clustering"""
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], c='blue', edgecolor='k', s=50, alpha=0.6)
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Plot data
if __name__ == '__main__':
    # Generate and plot each dataset
    print("Generating datasets for visualization...\n")
    
    # Dataset 1: Small
    data_small = generate_data()
    plot_raw_data(data_small, "Dataset 1: Small (70 points)")
    
    # Dataset 2: Sparse (70% noise)
    data_sparse = generate_sparse_data(n_points=1000, noise_ratio=0.7)
    plot_raw_data(data_sparse, "Dataset 2: Sparse (1000 points, 70% noise)")
    
    # Dataset 3: Dense (10 clusters)
    data_dense = generate_dense_data(n_points=1000, n_clusters=10)
    plot_raw_data(data_dense, "Dataset 3: Dense (1000 points, 10 clusters)")
    
    # Dataset 4: Large sparse (80% noise)
    data_large = generate_sparse_data(n_points=1500, noise_ratio=0.8)
    plot_raw_data(data_large, "Dataset 4: Large Sparse (1500 points, 80% noise)")
    
    print("All datasets plotted. Ready for benchmarking!")


if __name__ == '__main__':
    # Test 1: Small dataset 
    print("TEST 1:")
    data_small = generate_data()
    benchmark("Small Dataset", data_small, eps=1.0, minPts=5)
    
    # Test 2: Sparse dataset
    print("TEST 2:")
    data_sparse = generate_sparse_data(n_points=1000, noise_ratio=0.7)
    benchmark("Sparse Dataset", data_sparse, eps=1.0, minPts=5)
    
    # Test 3: Dense dataset 
    print("TEST 3:")
    data_dense = generate_dense_data(n_points=1000, n_clusters=10)
    benchmark("Dense Dataset", data_dense, eps=1.0, minPts=5)
    
    # Test 4: Large sparse dataset
    print("TEST 4:")
    data_large = generate_sparse_data(n_points=1500, noise_ratio=0.8)
    benchmark("Large Sparse Dataset", data_large, eps=1.0, minPts=5)

# def DBSCAN_Optimized_Grid(DB, distFunc, eps, minPts):
#     """
#     Optimized DBSCAN using GRID-BASED density estimation.
#     Much faster than MaxRS approach.
#     """
#     labels = {tuple(P): None for P in DB}
#     C = 0
#     unlabeled = set(tuple(P) for P in DB)

#     # Counters (DEBUG)
#     outer_loop_iterations = 0
#     grid_calls = 0
#     range_queries = 0

#     # Outer Loop: Repeatedly find densest region
#     while unlabeled:
#         outer_loop_iterations += 1
#         unlabeled_list = list(unlabeled)

#         # Find densest region using GRID (much faster than MaxRS!)
#         densest_point, max_density, dense_points = find_densest_region_grid(
#             unlabeled_list, eps, minPts
#         )
#         grid_calls += 1

#         # CASE 1: No dense region found, terminate
#         if max_density < minPts:
#             for P in unlabeled:
#                 labels[P] = -1
#             break

#         # CASE 2: Found potential cluster
#         # Use the densest point as seed
#         P = tuple(densest_point)

#         # Verify P is a core point
#         N = RangeQuery(DB, distFunc, P, eps)
#         range_queries += 1

#         # P doesn't have enough neighbors, mark as noise
#         if len(N) < minPts:
#             labels[P] = -1
#             unlabeled.discard(P)
#             continue

#         # P is a core point, expand cluster
#         C += 1
#         labels[P] = C
#         unlabeled.discard(P)

#         # Seed set - all neighbors except P
#         S = set(N) - {P}

#         while S:
#             Q = S.pop()
#             if labels[Q] == -1:
#                 labels[Q] = C
#             if labels[Q] is not None:
#                 continue
#             labels[Q] = C
#             unlabeled.discard(Q)
#             N = RangeQuery(DB, distFunc, Q, eps)
#             range_queries += 1
#             if len(N) >= minPts:
#                 S.update(N)

#     return labels, outer_loop_iterations, grid_calls, range_queries

# MAXRS (CARSON)
# def maxrs_sweepline(points, rect_w, rect_h):
#     """
#     Sweep line version.
    
#     Key optimization: For each x-position, filter points ONCE,
#     then sweep through y-positions checking only filtered points.
    
#     Args:
#         points: list of (x, y, weight)
#         rect_w, rect_h: rectangle width and height
    
#     Returns:
#         (best_x, best_y, max_sum, points_in_rect)
#     """
#     if not points:
#         return (0, 0, 0, [])
    
#     has_weights = len(points[0]) == 3    # Check if points have weights

#     x_candidates = sorted(set([p[0] for p in points])) # sorted array for X points
    
#     best_sum = -np.inf
#     best_pos = None
#     best_points = []
    
#     # Start left side of grid at x's
#     # Output: X jumps [1,5, 6, 8, ...]
#     for x in x_candidates:

#         # Filter y points within x-range [x, x+rect_w] 
#         # Output: list of (y, weight)
#         if has_weights:
#             points_in_x_range = [
#                 (py, w) for (px, py, w) in points
#                 if x <= px <= x + rect_w
#             ]
#         else:
#             points_in_x_range = [
#                 (py, 1) for (px, py) in points  # Assign weight=1
#                 if x <= px <= x + rect_w
#             ]
        
#         if not points_in_x_range:
#             continue
        
#         # Sort y tuple points (p) by y-coordinate (p[0]) for sweeping
#         # points_in_x_range.sort(key=lambda, p: p[0])
#         y_candidates = sorted(set(p[0] for p in points_in_x_range)) 
        
#         # Sweep through y coordinates IN X RANGE bottom-to-top
#         for y in y_candidates:
#             total = sum(
#                 w for (py, w) in points_in_x_range
#                 if y <= py <= y + rect_h
#             )
            
#             if total > best_sum:
#                 best_sum = total
#                 best_pos = (x, y)
#                 # Add points in rectangle to list w list comprehension
#                 best_points = [
#                     p for p in points
#                     if x <= p[0] <= x + rect_w and y <= p[1] <= y + rect_h
#                 ]
                
    
#     return best_pos + (best_sum, best_points)

# MAXRS (LANCE) 
# def maxrs_sweepline_L(points, rect_w, rect_h):
#     """
#     Sweep line O(n² log n) version.
    
#     Key optimization: For each x-position, filter points ONCE,
#     then sweep through y-positions checking only filtered points.
    
#     Args:
#         points: list of (x, y, weight)
#         rect_w, rect_h: rectangle width and height
#     Returns:
#         (best_x, best_y, max_sum)
#     """
#     if not points:
#         return (0, 0, 0, [])
    
#     # Check if points have weights
#     has_weights = len(points[0]) == 3
    
#     # Normalize points to always have weights
#     if has_weights:
#         normalized_points = points
#     else:
#         normalized_points = [(x, y, 1) for (x, y) in points]
    
#     # Points are tuples, which are sorted by first value by default
#     points_by_x_val = sorted(points)
    
#     best_sum = -np.inf
#     best_pos = None
#     best_points = []

#     # Add first element to list
#     points_in_x_range = [points_by_x_val[0]]
#     next_element_index = 1

#     # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
#     all_points_covered = False
    
#     # Start left side of grid at x's
#     # Output: X jumps [1,5, 6, 8, ...]
#     for p in points_by_x_val:
#         horizontal_ub = p[0] + rect_w
#         vertical_lb = p[1] - rect_h

#         # (1) Add new points to list
#         while next_element_index < len(points_by_x_val) and points_by_x_val[next_element_index][0] <= horizontal_ub:
#             points_in_x_range.append(points_by_x_val[next_element_index])
#             next_element_index += 1

#         # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
#         if next_element_index == len(points_by_x_val):
#             all_points_covered = True

#         # (2) Sort list by y-values
#         # THEORY: I think this is very fast outside the first run because the list is already mostly sorted
#         points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

#         # (3) Iterate until we reach a point in range vertically
#         i = 0
#         if not all_points_covered:
#             while i < len(points_in_x_range) and points_in_x_range[i][1] < vertical_lb:
#                 i += 1

#         # (4) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
#         current_sum = 0
#         current_pos = (p[0], points_in_x_range[i][1])
#         j = i
#         while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
#             current_sum += points_in_x_range[j][2]
#             j += 1

#         if current_sum > best_sum:
#             best_sum = current_sum
#             best_pos = current_pos
#             # Collect points in best rectangle
#             best_points = [
#                 pt for pt in points  # Use original points list
#                 if best_pos[0] <= pt[0] <= best_pos[0] + rect_w 
#                 and best_pos[1] <= pt[1] <= best_pos[1] + rect_h
#             ]
        
#         # (5) Iterate through points, altering window as we go, until we use the current left-most point as bottom
#         while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
#             # Remove weight of previous bottom point
#             current_sum -= points_in_x_range[i][2]
#             # Shift i to next point and reset current pos
#             i += 1
#             current_pos = (p[0], points_in_x_range[i][1])
#             # Shift j and add new points
#             while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
#                 current_sum += points_in_x_range[j][2]
#                 j += 1
#             # Check new window
#             if current_sum > best_sum:
#                 best_sum = current_sum
#                 best_pos = current_pos

#         # (6) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
#         if not all_points_covered:
#             while i < len(points_in_x_range) and points_in_x_range[i] != p:
#                 i += 1
#             if i < len(points_in_x_range):
#                 del points_in_x_range[i]
#         else:
#             break
    
#     return best_pos + (best_sum, best_points)

# GRID APPROX. APPROACH
# def find_densest_region_grid(points, eps, minPts):
#     """
#     Fast grid-based density estimation.
#     Much faster than MaxRS for finding dense regions.
    
#     Args:
#         points: list of tuples (x, y) or (x, y, w)
#         eps: DBSCAN epsilon parameter
#         minPts: minimum points threshold
    
#     Returns:
#         (densest_point, max_density, dense_points_list)
#     """
#     if not points:
#         return (None, 0, [])
    
#     # Use grid cells of size eps/2 
#     grid_size = eps / 2
#     grid = {}
    
#     # Populate grid
#     for p in points:
#         x, y = p[0], p[1]
#         cell = (int(x / grid_size), int(y / grid_size))
#         if cell not in grid:
#             grid[cell] = []
#         grid[cell].append(p)
    
#     # Find densest cell + neighbors
#     max_density = 0
#     densest_point = None
#     dense_points = []
    
#     for cell, cell_points in grid.items():
#         # Count points in cell + 8 neighbors (3x3)
#         local_points = list(cell_points)
        
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if dx == 0 and dy == 0:
#                     continue  # Already counted
#                 neighbor = (cell[0] + dx, cell[1] + dy)
#                 if neighbor in grid:
#                     local_points.extend(grid[neighbor])
        
#         density = len(local_points)
        
#         if density > max_density:
#             max_density = density
#             densest_point = cell_points[0]  # Pick any point from densest cell
#             dense_points = local_points
    
#     return (densest_point, max_density, dense_points)
