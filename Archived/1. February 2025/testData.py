import numpy as np
import matplotlib.pyplot as plt

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

    
if __name__ ==  "__main__":
    data = generate_data()
    
    # Plot the raw data
    plt.scatter(data[:, 0], data[:, 1], color='black', label='Raw Data')
    plt.legend()
    plt.show()