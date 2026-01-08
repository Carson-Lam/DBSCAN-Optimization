import numpy as np
import matplotlib.pyplot as plt

def maxrs(points, rect_w, rect_h):
    """
    Rudimentary MaxRS (Maximum Range Sum) demo.
    Brute-force scan of candidate rectangles.
    
    Args:
        points: list of (x, y, weight)
        rect_w, rect_h: rectangle width and height
        step: grid step for candidate rectangle positions

    Returns:
        (best_x, best_y, max_sum)

    """
    # min_x, max_x = min(xs), max(xs)
    # min_y, max_y = min(ys), max(ys)


    # Array for X points
    # Array for y points
    xs = [p[0] for p in points] 
    ys = [p[1] for p in points] 

    x_candidates = sorted(set(xs)) # Array of x candidates
    y_candidates = sorted(set(ys)) # Array of y  candidates
    
    best_sum = -np.inf
    best_pos = None

    # grid scan
    for x in x_candidates: #Start left side of grid at x coordinates 
        for y in y_candidates: #Define top left as arange of y's
            total = sum( #Define total points inside best grid
                w for (px, py, w) in points
                if x <= px <= x + rect_w and y <= py <= y + rect_h
            )
            if total > best_sum: #Redefine grid if new total found during grid scan
                best_sum = total
                best_pos = (x, y)

    return best_pos + (best_sum,)


# Points (x, y, weight)
points = [
    (2, 3, 5),
    (5, 4, 3),
    (3, 8, 4),
    (6, 7, 3),
    (7, 2, 2),
    (9, 6, 4),
    (3, 7, 5),
    (1, 5, 2),
    (6, 2, 8),
    (4, 9, 3),
    (8, 1, 6),
    (2, 4, 1),
    (7, 6, 7)
]

# Rectangle size
rect_w, rect_h = 3, 3

# Run demo
best_x, best_y, best_sum = maxrs(points, rect_w, rect_h)
print(f"Best rectangle at ({best_x:.2f}, {best_y:.2f}) with total weight = {best_sum}")

# --- Visualization ---
fig, ax = plt.subplots()
ax.scatter([p[0] for p in points], [p[1] for p in points],
           s=[p[2]*20 for p in points], color='blue', alpha=0.6, label='Points')

rect = plt.Rectangle((best_x, best_y), rect_w, rect_h,
                     edgecolor='red', facecolor='none', lw=2, label='Best rectangle')
ax.add_patch(rect)

ax.set_title("Rudimentary MaxRS Demo")
ax.legend()
plt.show()
