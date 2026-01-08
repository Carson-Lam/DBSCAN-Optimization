import numpy as np
import matplotlib.pyplot as plt
import time

# MAXRS
def maxrs_sweepline(points, rect_w, rect_h):
    """
    Sweep line version.
    
    Key optimization: For each x-position, filter points ONCE,
    then sweep through y-positions checking only filtered points.
    
    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
    
    Returns:
        (best_x, best_y, max_sum, points_in_rect)
    """
    if not points:
        return (0, 0, 0, [])
    
    has_weights = len(points[0]) == 3    # Check if points have weights

    x_candidates = sorted(set([p[0] for p in points])) # sorted array for X points
    
    best_sum = -np.inf
    best_pos = None
    best_points = []
    
    # Start left side of grid at x's
    # Output: X jumps [1,5, 6, 8, ...]
    for x in x_candidates:

        # Filter y points within x-range [x, x+rect_w] 
        # Output: list of (y, weight)
        if has_weights:
            points_in_x_range = [
                (py, w) for (px, py, w) in points
                if x <= px <= x + rect_w
            ]
        else:
            points_in_x_range = [
                (py, 1) for (px, py) in points  # Assign weight=1
                if x <= px <= x + rect_w
            ]
        
        if not points_in_x_range:
            continue
        
        # Sort y tuple points (p) by y-coordinate (p[0]) for sweeping
        # points_in_x_range.sort(key=lambda, p: p[0])
        y_candidates = sorted(set(p[0] for p in points_in_x_range)) 
        
        # Sweep through y coordinates IN X RANGE bottom-to-top
        for y in y_candidates:
            total = sum(
                w for (py, w) in points_in_x_range
                if y <= py <= y + rect_h
            )
            
            if total > best_sum:
                best_sum = total
                best_pos = (x, y)
                # Add points in rectangle to list w list comprehension
                best_points = [
                    p for p in points
                    if x <= p[0] <= x + rect_w and y <= p[1] <= y + rect_h
                ]
                
    
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

def DBSCAN_Optimized(DB, distFunc, eps, minPts):
    """
    Optimized DBSCAN using MaxRS to find densest regions first.
    Uses Option 1: Select any point inside M' as seed.
    """
    # labels = {P: None for P in DB} #Tracks labels for each point ---  GENERIC
    labels = {tuple(P): None for P in DB} # Convert P to tuple for hashing 
    C = 0
    unlabeled = set(tuple(P) for P in DB)

    # Calculate rectangle sizes
    rect_w1, rect_h1 = eps * np.sqrt(2), eps * np.sqrt(2) 
    rect_w2, rect_h2 = 2 * eps, 2 * eps

    # Outer Loop: Repeated find densest region w MaxRS
    while unlabeled: 
        unlabedList = list(unlabeled)

        # M - Smaller MaxRS rect (inside DBSCAN circle) 
        x1, y1, sum1, points1 = maxrs_sweepline(unlabedList, rect_w1, rect_h1)

        # M' - Larger MaxRS rect (outside DBSCAN circle)
        x2, y2, sum2, points2 = maxrs_sweepline(unlabedList, rect_w2, rect_h2)


        # CASE 1: |M'| < minpoints so terminate DBSCAN, return clusters, everything else noise
        if sum2 < minPts:
            for P in unlabeled:
                labels[P] = -1
            break
        
        # CASE 2: there may be a cluster
        # Option 1: Take any point inside M' as seed point P
        P = tuple(points2[0])

        # Check if P is a core point by finding it's neighbors
        N = RangeQuery(DB, distFunc, P, eps) 

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
            if len(N) >= minPts:
                S.update(N)

    return labels

def DBSCAN(DB, distFunc, eps, minPts):
    # labels = {P: None for P in DB} #Tracks labels for each point ---  GENERIC
    labels = {tuple(P): None for P in DB} # Convert P to tuple for hashing 
    C = 0

    for P in DB:
        P_tuple = tuple(P) # Convert P to tuple for hashing 
        if labels[P_tuple] is not None: #If point labeled, continue
            continue 

        #Obtain amount of points in euclidian distance (ε radius)
        N = RangeQuery(DB, distFunc, P, eps) 

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
            if len(N) >= minPts:
                S.update(N)

    return labels

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
    times_orig = []
    for _ in range(runs):
        start = time.time()
        labels_orig = DBSCAN(data, euclidean_distance, eps, minPts)
        times_orig.append(time.time() - start)
    
    avg_orig = np.mean(times_orig)
    clusters_orig = len(set(labels_orig.values()) - {-1, None})
    noise_orig = sum(1 for v in labels_orig.values() if v == -1)
    
    # Optimized DBSCAN
    times_opt = []
    for _ in range(runs):
        start = time.time()
        labels_opt = DBSCAN_Optimized(data, euclidean_distance, eps, minPts)
        times_opt.append(time.time() - start)
    
    avg_opt = np.mean(times_opt)
    clusters_opt = len(set(labels_opt.values()) - {-1, None})
    noise_opt = sum(1 for v in labels_opt.values() if v == -1)
    
    # Results
    print(f"\nOriginal DBSCAN:")
    print(f"  Avg Runtime: {avg_orig:.6f}s (±{np.std(times_orig):.6f}s)")
    print(f"  Clusters: {clusters_orig}, Noise: {noise_orig}")
    
    print(f"\nOptimized DBSCAN:")
    print(f"  Avg Runtime: {avg_opt:.6f}s (±{np.std(times_opt):.6f}s)")
    print(f"  Clusters: {clusters_opt}, Noise: {noise_opt}")
    
    speedup = avg_orig / avg_opt
    if speedup > 1:
        print(f"\n✓ Optimized is {speedup:.2f}x FASTER")
    else:
        print(f"\n✗ Optimized is {1/speedup:.2f}x SLOWER")
    
    return labels_orig, labels_opt

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
    data_large = generate_sparse_data(n_points=5000, noise_ratio=0.8)
    benchmark("Large Sparse Dataset", data_large, eps=1.0, minPts=5)


# Plot data
# if __name__ == '__main__':
    # # Generate and plot each dataset
    # print("Generating datasets for visualization...\n")
    
    # # Dataset 1: Small
    # data_small = generate_data()
    # plot_raw_data(data_small, "Dataset 1: Small (70 points)")
    
    # # Dataset 2: Sparse (70% noise)
    # data_sparse = generate_sparse_data(n_points=1000, noise_ratio=0.7)
    # plot_raw_data(data_sparse, "Dataset 2: Sparse (1000 points, 70% noise)")
    
    # # Dataset 3: Dense (10 clusters)
    # data_dense = generate_dense_data(n_points=1000, n_clusters=10)
    # plot_raw_data(data_dense, "Dataset 3: Dense (1000 points, 10 clusters)")
    
    # # Dataset 4: Large sparse (80% noise)
    # data_large = generate_sparse_data(n_points=5000, noise_ratio=0.8)
    # plot_raw_data(data_large, "Dataset 4: Large Sparse (5000 points, 80% noise)")
    
    # print("All datasets plotted. Ready for benchmarking!")