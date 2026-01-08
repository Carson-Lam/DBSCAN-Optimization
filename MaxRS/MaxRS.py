import numpy as np
import matplotlib.pyplot as plt
import time

def maxrs_sweepline(points, rect_w, rect_h):
    """
    Sweep line O(n² log n) version.
    
    Key optimization: For each x-position, filter points ONCE,
    then sweep through y-positions checking only filtered points.
    
    Args:
        points: list of (x, y) tuples OR (x, y, weight) tuples
        rect_w, rect_h: rectangle width and height
    Returns:
        (best_x, best_y, max_sum)
    """
    if not points:
        return (0, 0, 0)
    
    # Check if points have weights
    has_weights = len(points[0]) == 3
    
    x_candidates = sorted(set([p[0] for p in points])) # sorted array for X points
    
    best_sum = -np.inf
    best_pos = None
    
    # Start left side of grid at x's
    # Output: X jumps [1,5, 6, 8, ...]
    for x in x_candidates:

        # Filter y points within x-range [x, x+rect_w] 
        # Output: list of (y, weight)
        points_in_x_range = [
            (py, w) for (px, py, w) in points
            if x <= px <= x + rect_w
        ]
        
        if not points_in_x_range:
            continue
        
        # Sort y tuple points (p) by y-coordinate (p[0]) for sweeping
        # Sort syntax: list.sort(key = ???, reverse = ???)
        # points_in_x_range.sort(key=lambda p: p[0])
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
    
    return best_pos + (best_sum,)

# Randomly generated Test data, n = 100
np.random.seed(3)
n = 100
points = [(np.random.uniform(0, 50), 
           np.random.uniform(0, 50), 
           np.random.randint(1, 10)) 
          for _ in range(n)]

rect_w, rect_h = 5, 5

print(points)

print("="*60)
print("MaxRS Demo")
print("="*60)

start = time.time()
bx2, by2, bs2 = maxrs_sweepline(points, rect_w, rect_h)
time2 = time.time() - start

print(f"Result: ({bx2:.2f}, {by2:.2f}) with sum = {bs2}")
print(f"Time: {time2*1000:.4f} ms") # Convert to milliseconds

# ============================================
# Visualization
# ============================================
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter([p[0] for p in points], [p[1] for p in points],
           s=[p[2]*30 for p in points], color='blue', alpha=0.6, 
           label='Points (size = weight)', zorder=3)

for i, (x, y, w) in enumerate(points):
    ax.annotate(f'{w}', (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.7, fontweight='bold')

# Show rectangles
rect = plt.Rectangle((bx2, by2), rect_w, rect_h,
                     edgecolor='blue', facecolor='blue', 
                     alpha=0.2, lw=3, label=f'Rectangle sweep (sum={bs2})')
ax.add_patch(rect)

# Show candidate grid lines
for x in set(p[0] for p in points):
    ax.axvline(x, color='gray', alpha=0.15, linestyle='--', linewidth=0.5)
for y in set(p[1] for p in points):
    ax.axhline(y, color='gray', alpha=0.15, linestyle='--', linewidth=0.5)

ax.set_title(f"MaxRS Sweep Line Algorithm\n" + 
             f"Optimal position (w sweep): ({bx2}, {by2}) | Total weight: {bs2}",
             fontsize=13, fontweight='bold')

ax.set_xlabel('X coordinate', fontsize=11)
ax.set_ylabel('Y coordinate', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.5)

plt.tight_layout()
plt.show()