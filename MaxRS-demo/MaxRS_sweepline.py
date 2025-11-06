import numpy as np
import matplotlib.pyplot as plt
import time

def maxrs(points, rect_w, rect_h, step=1.0):
    """
    Original O(n³) version - checks ALL points for every (x,y) pair.

    Brute force grid scan, moves a rectangle of size (rect_w, rect_h) with step size 'step'

    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
        Step: step size for grid scan
    Returns:
        (best_x, best_y, max_sum)
    """
    xs = [p[0] for p in points] # Array for X points
    ys = [p[1] for p in points] # Array for y points
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    best_sum = -np.inf
    best_pos = None

    # Slides grid using np.arange(min_x, max_x, step) with step size
    # Example: X jumps [min_x, min_x+step, min_x+(2*step), ...]
    for x in np.arange(min_x, max_x, step):

        for y in np.arange(min_y, max_y, step):
            total = sum( 
                w for (px, py, w) in points
                if x <= px <= x + rect_w and y <= py <= y + rect_h
            )
            if total > best_sum: #Redefine grid if new total found during grid scan
                best_sum = total
                best_pos = (x, y)

    return best_pos + (best_sum,)

def maxrs_optimized(points, rect_w, rect_h):
    """
    Original O(n³) version - checks ALL points for every (x,y) pair.

    Slightly optimized grid scan, starts rectangle's left edge at sorted x's

    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
    Returns:
        (best_x, best_y, max_sum)
    """
    x_candidates = sorted(set([p[0] for p in points])) # sorted array for X points
    y_candidates = sorted(set([p[1] for p in points])) # Sorted array for Y points
    
    best_sum = -np.inf
    best_pos = None

    # Start left side of grid at x's
    # Example: X jumps [1,5, 6, 8, ...]
    for x in x_candidates: 

        for y in y_candidates: 
            total = sum(
                w for (px, py, w) in points
                if x <= px <= x + rect_w and y <= py <= y + rect_h
            )
            if total > best_sum: #Redefine grid if new total found during grid scan
                best_sum = total
                best_pos = (x, y)

    return best_pos + (best_sum,)


def maxrs_sweepline(points, rect_w, rect_h):
    """
    Sweep line O(n² log n) version.
    
    Key optimization: For each x-position, filter points ONCE,
    then sweep through y-positions checking only filtered points.
    
    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
    Returns:
        (best_x, best_y, max_sum)
    """
    if not points:
        return (0, 0, 0)
    
    x_candidates = sorted(set([p[0] for p in points])) # sorted array for X points
    
    best_sum = -np.inf
    best_pos = None
    
    # Start left side of grid at x's
    # Example: X jumps [1,5, 6, 8, ...]
    for x in x_candidates:

        # Filter y points within x-range [x, x+rect_w] 
        # Example: list of (y, weight)
        points_in_x_range = [
            (py, w) for (px, py, w) in points
            if x <= px <= x + rect_w
        ]
        
        if not points_in_x_range:
            continue
        
        # Sort y tuple points (p) by y-coordinate (p[0]) for sweeping
        # Sort syntax: .sort(key = ???, reverse = ???)
        points_in_x_range.sort(key=lambda p: p[0])
        
        y_candidates = sorted(set(p[0] for p in points_in_x_range)) # sorted array for y points IN X RANGE
        
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


# Hardcoded Test data, n = 13
# points = [
#     (2, 3, 5),
#     (5, 4, 3),
#     (3, 8, 4),
#     (6, 7, 3),
#     (7, 2, 2),
#     (9, 6, 4),
#     (3, 7, 5),
#     (1, 5, 2),
#     (6, 2, 8),
#     (4, 9, 3),
#     (8, 1, 6),
#     (2, 4, 1),
#     (7, 6, 7)
# ]

# Randomly generated Test data, n = 100
# np.random.seed(42)
# n = 100
# points = [(np.random.uniform(0, 50), 
#            np.random.uniform(0, 50), 
#            np.random.randint(1, 10)) 
#           for _ in range(n)]

# rect_w, rect_h = 3, 3

print("="*60)
print("COMPARING BASIC vs OPTIMIZED VS SWEEP LINE")
print("="*60)

print("\n[1] OPTIMIZEED O(n³) VERSION")
start = time.time()
bx1, by1, bs1 = maxrs_optimized(points, rect_w, rect_h)
time1 = time.time() - start
print(f"    Result: ({bx1:.2f}, {by1:.2f}) with sum = {bs1}")
print(f"    Time: {time1*1000:.4f} ms") # Convert to milliseconds

print("\n[2] SWEEP LINE O(n² log n) VERSION")
start = time.time()
bx2, by2, bs2 = maxrs_sweepline(points, rect_w, rect_h)
time2 = time.time() - start
print(f"    Result: ({bx2:.2f}, {by2:.2f}) with sum = {bs2}")
print(f"    Time: {time2*1000:.4f} ms") # Convert to milliseconds

# Verify results match using sum bs1 and bs2
print("\n" + "="*60)
if bs1 == bs2:
    print("✓ Both versions produce the same result!")
    if time1 > time2:
        print(f"✓ Sweep line is {time1/time2:.2f}x faster!")
else:
    print("✗ Results don't match - bug in implementation!")
print("="*60)

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter([p[0] for p in points], [p[1] for p in points],
           s=[p[2]*30 for p in points], color='blue', alpha=0.6, 
           label='Points (size = weight)', zorder=3)

for i, (x, y, w) in enumerate(points):
    ax.annotate(f'{w}', (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.7, fontweight='bold')

rect = plt.Rectangle((bx2, by2), rect_w, rect_h,
                     edgecolor='red', facecolor='red', 
                     alpha=0.2, lw=3, label=f'Best rectangle (sum={bs2})')
ax.add_patch(rect)

# Show candidate grid lines

# 6/11/2025: Candidates line up w grid, so this is currently useless

# for x in set(p[0] for p in points):
#     ax.axvline(x, color='gray', alpha=0.15, linestyle='--', linewidth=0.5)
# for y in set(p[1] for p in points):
#     ax.axhline(y, color='gray', alpha=0.15, linestyle='--', linewidth=0.5)

ax.set_title(f"MaxRS Sweep Line Algorithm\n" + 
             f"Optimal position: ({bx2}, {by2}) | Total weight: {bs2}",
             fontsize=13, fontweight='bold')

ax.set_xlabel('X coordinate', fontsize=11)
ax.set_ylabel('Y coordinate', fontsize=11)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()