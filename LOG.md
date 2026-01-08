## ADDITIONS 7/01/2026:
- Changed MaxRS to also return list of pts inside rect --> now returns (bx1, bx2, sum, list of pts in recct)

- Changed MaxRS to handle unweighted pts too --> now can take both (x, y) and (x, y, w) tuple 

- Implemented DBSCAN_Optimized following Pseudocode outline

- Added three different datasets & a benchmark that compares DBSCAN_Optimized with DBSCAN.
<br>    --> Small dataset
<br>    --> Sparse dataset
<br>    --> Dense dataset
<br>    --> Large Sparse dataset 

## RESULTS:
- TEST 1: Small Dataset (70 points)
<br> Optimized is 1.80x FASTER -- 0.073923s vs 0.041018s

- TEST 2: Sparse Dataset (1000 points) 
<br> ✗ Optimized is 5.15x SLOWER -- 38.990075s vs 7.56857s

- TEST 3: Dense Dataset (1000 points)
<br> ✗ Optimized is 3.33x SLOWER -- 26.912178s vs 8.077683s

- TEST 4: Large Sparse Dataset (5000 points)

## NEXT STEPS:
**Issue:** MaxRS-Sweepline runs in O(n^2 log n), which is higher
than default DBSCAN of O(n^2). This means for k points, we do
O(k * n^2 logn) instead of O(k*n logn)

