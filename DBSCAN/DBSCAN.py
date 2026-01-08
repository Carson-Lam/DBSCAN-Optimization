import numpy as np
import matplotlib.pyplot as plt
import time

## DBSCAN ##
## Labels dataset with label P grouped by an increasing value C

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

if __name__ == '__main__':
    # Generate data
    data = generate_data()

    # Define DBSCAN Parameters
    eps = 1.0
    minPts = 5

    # Run DBSCAN with timing
    start_time = time.time()
    labels = DBSCAN(data, euclidean_distance, eps, minPts)
    end_time = time.time()
    
    print(f"Original DBSCAN Runtime: {end_time - start_time:.6f} seconds")
    print(f"Clusters found: {len(set(labels.values()) - {-1, None})}")
    print(f"Noise points: {sum(1 for v in labels.values() if v == -1)}")
    
    label_values = [labels[tuple(point)] for point in data]
    
    # Plot results
    plt.scatter(data[:, 0], data[:, 1], c=label_values, cmap='tab20', edgecolor='k')
    plt.title("Original DBSCAN Clustering Results")
    plt.show()
