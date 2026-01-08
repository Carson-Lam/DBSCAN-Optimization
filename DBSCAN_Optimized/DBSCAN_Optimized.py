# Run (two) MaxRS *on all the points without a cluster (or noise) label so far* with a square that has a side length of eps*sqrt(2) (and side length of 2eps)
# Let M (M') be the returned maxRS rectangle 
# Case 1:
#   If  |M'| <minpoints then terminate all of DBSCAN and return the clusters so far. Everything else must be noise
# Case 2:  
#   Otherwise, we know there may be a cluster:
#       Option 1: Take any point inside M' as P
#       Option 2: Create a new "dummy" point at the center of M' as P.  (this point will not be added to a cluster because it doesn't really exist)
#   If the resulting cluster has only border points, then turn them back to noise (that's the case where the dummy point connected noise points into a cluster)

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

if __name__ == '__main__':
    # Generate data 
    data = generate_data()

    # Define DBSCAN Parameters
    eps = 1.0
    minPts = 5

    # Run Optimized DBSCAN with timing
    start_time = time.time()
    labels = DBSCAN_Optimized(data, euclidean_distance, eps, minPts)
    end_time = time.time()

    print(f"Optimized DBSCAN Runtime: {end_time - start_time:.6f} seconds")
    print(f"Clusters found: {len(set(labels.values()) - {-1, None})}")
    print(f"Noise points: {sum(1 for v in labels.values() if v == -1)}")
    label_values = [labels[tuple(point)] for point in data] 
    
    # Plot results
    plt.scatter(data[:, 0], data[:, 1], c=label_values, cmap='tab20', edgecolor='k')
    plt.title("DBSCAN Clustering Results")
    plt.show()



