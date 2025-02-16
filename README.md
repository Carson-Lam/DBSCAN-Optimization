# DBSCAN Optimization (Density Based algorithm for discovering clusters)

## Overview

In partnership with Andreas Zuefle [[1]](https://www.zuefle.org/spatial-clustering), this repository is an implementation for a proposed optimization of the largely popular DBSCAN [[2]](https://dl.acm.org/doi/abs/10.5555/3001460.3001507). This optimization aims to improve the time complexity of DBSCAN (which runs in O(n^2) time without index support and O(n log n) with indexing) by incorporating result-sensitivity.

## Methodology

Our goal is to incorporate methodologies used in solving the ACM Maximum Range Sum (MaxRS) Problem [[3]](https://dl.acm.org/doi/abs/10.14778/2350229.2350230) [[4]](https://dl.acm.org/doi/abs/10.14778/2536258.2536266) into DBSCAN. 
- These methodologies currently run in O(nlogn) runtime.

 Instead of iterating through each point randomly, DBSCAN will visit points descending by cluster density and **terminate** when the next densest cluster does not meet the minimum DBSCAN density level. 

## Runtime
Since each MaxRS process to find the densest area in a space runs in O(nlogn), the optimized DBSCAN algorithm should have **O(knlogn)** runtime, where k constitutes the amount of points in clusters.

## Download and Run
A trial set of data (visualized with matplotlib) has been provided as a testable implementation of DBScan. 

1. Download the *algorithm* folder.

2. The raw data can be plotted by running *testData.py*

3. The cluster color coordinated data can be plotted by running *testScan.py*

## Project Dependencies

- [numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)

## References

[1] Zuefle, Andreas. “*Spatial Clustering*” Zuefle.org, Retrieved January 15th, 2024 from [https://www.zuefle.org/spatial-clustering](https://www.zuefle.org/spatial-clustering)

[2] Ester, M., Kriegel, H.P., Sander, J. and Xu, X., 1996, August. A density-based algorithm for discovering clusters in large spatial databases with noise. In kdd (Vol. 96, No. 34, pp. 226-231). 

[3] Choi, D.W., Chung, C.W. and Tao, Y., 2012. A scalable algorithm for maximizing range sum in spatial databases. Proceedings of the VLDB Endowment, 5(11), pp.1088-1099. 

[4] Tao, Y., Hu, X., Choi, D.W. and Chung, C.W., 2013, August. Approximate MaxRS in spatial databases. In 39th International Conference on Very Large Data Bases 2013, VLDB 2013 (Vol. 6, pp. 1546-1557). International Conference on Very Large Data Bases. 
