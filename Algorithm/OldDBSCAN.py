import numpy as np
def RangeQuery(DB, distFunc, Q, eps): #Where epsilon ε defines the minimum size of a cluster
    N = []
    for P in DB:
        if distFunc(Q, P) <= eps:
            N.append(P)
        
    return N

def euclidean_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def DBSCAN(DB, distFunc, eps, minPts):
    labels = {P: None for P in DB} #Tracks labels for each point
    C = 0
    for P in DB:
        if labels[P] is not None: #If point labeled, continue
            continue 

        #Obtain amount of points in euclidian distance (ε radius)
        N = RangeQuery(DB, distFunc, P, eps) 

        if len(N) < minPts:
            labels[P] = -1 #Mark as noise (-1)
            continue

        C += 1 #Move onto next point in cluster for labeling
        labels[P] = C  #Label initial cluster point  
        S = set(N) - {P} #Set a new set for the cluster excluding initial point

        while S:
            Q = S.pop()
            if labels[Q] == -1: labels[Q] = C #Change from noise to border
            if labels[Q] is not None: continue #if already labeled, exit
            labels[Q] = C #Label neighbor
            N = RangeQuery(DB, distFunc, Q, eps)
            if len(N) >= minPts:
                S.update(N)

    return labels


