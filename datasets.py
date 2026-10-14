"""
datasets.py
=============
Incremental grid-based seed selection
"""
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os


def generate_synthetic(n_total, alpha, n_clusters, stdev, seed=42):
    """
    Generate a synthetic 2D point set in [0, 1] x [0, 1].

    Parameters
    ----------
    n_total    : total number of points
    alpha      : noise fraction (e.g. 0.90 → 90% noise)
    n_clusters : number of Gaussian clusters
    stdev      : standard deviation of each cluster
    seed       : random seed for reproducibility

    Returns
    -------
    data       : np.ndarray of shape (n_total, 2), all points in [0,1]^2
    centers    : list of (x, y) cluster centroids
    """
    rng = np.random.default_rng(seed)

    n_noise        = int(n_total * alpha)
    n_cluster_pts  = n_total - n_noise
    pts_per_cluster = n_cluster_pts // n_clusters
    remainder       = n_cluster_pts - pts_per_cluster * n_clusters

    # Cluster centroids placed uniformly at random in [0.1, 0.9]
    # (margin avoids clusters right at the border)
    centers = rng.uniform(0.1, 0.9, size=(n_clusters, 2))

    cluster_arrays = []
    for i, center in enumerate(centers):
        # Last cluster gets any leftover points from integer division
        n_pts = pts_per_cluster + (remainder if i == n_clusters - 1 else 0)
        pts = rng.normal(loc=center, scale=stdev, size=(n_pts, 2))
        # Clip to [0, 1] to keep all points in the unit square
        pts = np.clip(pts, 0.0, 1.0)
        cluster_arrays.append(pts)

    # Noise: uniform over [0, 1]^2
    noise = rng.uniform(0.0, 1.0, size=(n_noise, 2))

    data = np.vstack(cluster_arrays + [noise])

    # Shuffle so cluster points and noise are not ordered
    idx = rng.permutation(len(data))
    data = data[idx]

    return data, centers.tolist()

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


def load_osm_city(json_file, bbox):
    """Load an OSM restaurant JSON file and convert to km coordinates."""
    restaurants = load_osm_data(json_file)
    if restaurants is None:
        return None
    data = convert_to_xy(restaurants, bbox)
    print(f"  Loaded {len(data)} points from {json_file}")
    return data


def load_all_real_datasets():
    """
    Load Atlanta, NYC, and Chicago restaurant datasets.
    Run osm.py with the bounding boxes below to generate the JSON files.
    """
    datasets = {}

    atlanta_bbox = (33.6490, -84.5510, 33.8860, -84.2890)
    nyc_bbox     = (40.4774, -74.2591, 40.9176, -73.7004)
    chicago_bbox = (41.6445, -87.9401, 42.0230, -87.5240)

    specs = [
        ("Atlanta",  "atlanta_restaurants_osm.json",  atlanta_bbox),
        ("NYC",      "nyc_restaurants_osm.json",      nyc_bbox),
        ("Chicago",  "chicago_restaurants_osm.json",  chicago_bbox),
    ]

    for name, fname, bbox in specs:
        if os.path.exists(fname):
            data = load_osm_city(fname, bbox)
            if data is not None:
                datasets[name] = data
        else:
            print(f"  [SKIP] {fname} not found. Run osm.py with bbox={bbox}")

    return datasets

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

def gen_constant_density(n, noise_ratio=0.7, n_clusters=5, seed=1):
    rng = np.random.default_rng(seed)
    side = np.sqrt(n / 10.0)              # 10 points per unit^2
    n_noise = int(n * noise_ratio)
    n_clu = n - n_noise
    per = n_clu // n_clusters
    centers = rng.uniform(0.1 * side, 0.9 * side, size=(n_clusters, 2))
    parts = [rng.normal(c, 0.12, size=(per, 2)) for c in centers]
    parts.append(rng.uniform(0, side, size=(n - per * n_clusters, 2)))
    data = np.clip(np.vstack(parts), 0, side)
    return data[rng.permutation(len(data))]