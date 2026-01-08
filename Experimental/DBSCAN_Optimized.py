# Run (two) MaxRS *on all the points without a cluster (or noise) label so far* with a square that has a side length of eps*sqrt(2) (and side length of 2eps)
# Let M (M') be the returned maxRS rectangle 
# Case 1:
#   If  |M'| <minpoints then terminate all of DBSCAN and  nreturn the clusters so far. Everything else must be noise
# Case 2:  
#   Otherwise, we know there may be a cluster:
#       Option 1: Take any point inside M' as P
#       Option 2: Create a new "dummy" point at the center of M' as P.  (this point will not be added to a cluster because it doesn't really exist)
#   If the resulting cluster has only border points, then turn them back to noise (that's the case where the dummy point connected noise points into a cluster)

