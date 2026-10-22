import numpy as np
import matplotlib.pyplot as plt
import time
import heapq
from sortedcontainers import SortedList
from bintrees import AVLTree

def maxrs(points, rect_w, rect_h, step=1.0):
    # Let num points := n
    # Let num steps := m (assuming square)
    # Runtime = O(m^2*n)
    # This is guaranteed optimal if step size is half rectangle size or less
    # Runtime could be very fast here depending on m
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
    # Let num points := n
    # Sort time = 2*n*log(n)
    # Check time = n*n*n
    # Runtime = n^3 + 2n logn = O(n^3)
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
    # Let num points := n
    # Sort time = n*log(n)
    # For each candidate: n
    #   Generate points in x range: n
    #   Sort points in x range : m*log(m) where m <<< n (probably)
    #   For each point in x range: m
    #       Generate points in y range: m
    # Runtime = n*log(n) + n*(n + m*log(m) + m(m)) = n*log(n) + n^2 + n*m*log(m) + n*m^2 = O(n^2) assuming m^2 < n
    # Assuming small m, this could be very fast
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

# NOTE: Basically, the heap approach was poorly thought out, and this shouldn't work well
def maxrs_sweepline_2(points, rect_w, rect_h):
    # Let num points := n
    # Sort time = n*log(n)
    # For each candidate: n
    #   Generate points in x range and push on to heap: m*log(m) where m <<< n (probably)
    #   Unheap to sort them: m*log(m)
    #   For each point in x range: m
    #       Generate points in y range: m
    #   Reheap elements: m*log(m)
    # Runtime = n*log(n) + n*(3*m*log(m) + m(m)) = n*log(n) + 3*n*m*log(m) + n*m^2 = O(n*m^2)
    # Assuming small m, this could be very fast
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
    
    # Idea: Keep a heap containing the points currently within range horizontally
    #       sorted by their values vertically. Then we don't need to filter for
    #       candidates each time. In fact, I think we can immediately filter
    #       candidates even further by only considering points from y - rect_height
    #       to y as our "bottoms" of the rectangle

    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(points)

    best_sum = -np.inf
    best_pos = None
    
    # Create a heap to store the points in range, sorted by their y-values. Add first point
    heap = []
    heapq.heappush(heap, (points_by_x_val[0][1], points_by_x_val[0])) # Need to put y-value first for sorting
    next_element_to_heap = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    all_points_covered = False

    # Iterate from leftmost point
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h
        vertical_ub = p[1] + rect_h

        # (1) Add new points to heap
        while next_element_to_heap < len(points_by_x_val) and points_by_x_val[next_element_to_heap][0] <= horizontal_ub:
            heapq.heappush(heap, (points_by_x_val[next_element_to_heap][1], points_by_x_val[next_element_to_heap]))
            next_element_to_heap += 1

        # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
        if next_element_to_heap == len(points_by_x_val):
            all_points_covered = True
        
        # (2) Iterate until we reach a point in range vertically
        elements_below_range = []
        next_element = heap[0]
        if not all_points_covered:
            while next_element[0] < vertical_lb:
                elements_below_range.append(heapq.heappop(heap))
                next_element = heap[0]

        # (3) Now, iterate again until we reach a point out of range vertically
        elements_in_range = [] # This list will be sorted by y-value
        if not all_points_covered:
            while vertical_lb <= next_element[0] <= vertical_ub:
                elements_in_range.append(heapq.heappop(heap))
                if len(heap) > 0:
                    next_element = heap[0]
                else:
                    break
        else:
            while len(heap) > 0:
                elements_in_range.append(heapq.heappop(heap))

        # (4) Now, use each point as bottom of rectangle and check weight until we reach current point
        for i in range(len(elements_in_range)):
            # (5) take sum of points in range this way
            total = 0
            j = i
            while j < len(elements_in_range) and elements_in_range[j][0] <= elements_in_range[i][0] + rect_h:
                total += elements_in_range[j][1][2]
                j += 1

            if total > best_sum:
                best_sum = total
                best_pos = (p[0], elements_in_range[i][1][1])

            if elements_in_range[i][1] == p and not all_points_covered:
                break
        
        if all_points_covered:
            break
        
        # (6) remove the current element and reheap the others
        del elements_in_range[i]
        for el in elements_below_range:
            heapq.heappush(heap, el)
        for el in elements_in_range:
            heapq.heappush(heap, el)
    
    return best_pos + (best_sum,)

# NOTE: This should basically be an improved version of Carson's algorithm
def maxrs_sweepline_3(points, rect_w, rect_h):
    # Let num points := n
    # Sort time = n*log(n)
    # For each candidate: n
    #   Generate points in x range: m where m <<< n (probably)
    #   Sort points in x range : m*log(m)
    #   Run sliding window: m
    # Runtime = n*log(n) + n*(m + m*log(m) + m) = n*log(n) + 2*n*m + n*m*log(m) = O(n*m*log(m))
    # Assuming small m, this could be very fast
    # THEORY: I think this is very fast because the list is already mostly sorted after the first iteration, but unclear. Maybe sorting is closer to O(m) time then
    # According to ChatGPT, sorted uses timsort which is close to linear on already sorted data. In this case, time complexity should be O(n*m) !!!
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
    
    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(points)
    
    best_sum = -np.inf
    best_pos = None

    # Add first element to list
    points_in_x_range = [points_by_x_val[0]]
    next_element_index = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    all_points_covered = False
    
    # Start left side of grid at x's
    # Output: X jumps [1,5, 6, 8, ...]
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h

        # (1) Add new points to list
        while next_element_index < len(points_by_x_val) and points_by_x_val[next_element_index][0] <= horizontal_ub:
            points_in_x_range.append(points_by_x_val[next_element_index])
            next_element_index += 1

        # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
        if next_element_index == len(points_by_x_val):
            all_points_covered = True

        # (2) Sort list by y-values
        # THEORY: I think this is very fast outside the first run because the list is already mostly sorted
        points_in_x_range = sorted(points_in_x_range, key=lambda t: t[1])

        # (3) Iterate until we reach a point in range vertically
        i = 0
        if not all_points_covered:
            while points_in_x_range[i][1] < vertical_lb:
                i += 1

        # (4) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = current_pos
        
        # (5) Iterate through points, altering window as we go, until we use the current left-most point as bottom
        while j < len(points_in_x_range) and (points_in_x_range[i] != p or all_points_covered):
            # Remove weight of previous bottom point
            current_sum -= points_in_x_range[i][2]
            # Shift i to next point and reset current pos
            i += 1
            current_pos = (p[0], points_in_x_range[i][1])
            # Shift j and add new points
            while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
                current_sum += points_in_x_range[j][2]
                j += 1
            # Check new window
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = current_pos

        # (6) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
        if not all_points_covered:
            while points_in_x_range[i] != p:
                i += 1
            del points_in_x_range[i]
        else:
            break
    
    return best_pos + (best_sum,)

# NOTE: This should, in theory, be the fastest algorithm, but is failing dramatically
def maxrs_sweepline_4(points, rect_w, rect_h):
    # Let num points := n
    # Sort time = n*log(n)
    # For each candidate: n
    #   Generate points in x range and add to sorted list: m*log(m) where m <<< n (probably)
    #   Run sliding window: m
    # Runtime = n*log(n) + n*(m*log(m) + m) = n*log(n) + n*m*log(m) + n*m = O(n*m*log(m))
    # Assuming small m, this could be very fast
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
    
    # Idea: Keep a SortedList containing the points currently within range horizontally,
    #       sorted by their values vertically. Then we don't need to filter for candidates 
    #       each time. SortedList supports O(logn) insertion, O(1) random access for O(n)
    #       traversal, and O(logn) deletion (although in worst case, which is rare, deletion
    #       is O(nlogn))

    # Points are tuples, which are sorted by first value by default
    points_by_x_val = sorted(points)

    best_sum = -np.inf
    best_pos = None
    
    # Create a SortedList to store the points in range, sorted by their y-values. Add first point
    points_in_x_range = SortedList(key=lambda t: t[1]) # sort by y-value
    points_in_x_range.add(points_by_x_val[0])
    next_element_to_add = 1

    # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
    all_points_covered = False

    # Iterate from leftmost point
    for p in points_by_x_val:
        horizontal_ub = p[0] + rect_w
        vertical_lb = p[1] - rect_h
        vertical_ub = p[1] + rect_h

        # (1) Add new points to sorted list
        while next_element_to_add < len(points_by_x_val) and points_by_x_val[next_element_to_add][0] <= horizontal_ub:
            points_in_x_range.add(points_by_x_val[next_element_to_add])
            next_element_to_add += 1

        # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
        if next_element_to_add == len(points_by_x_val):
            all_points_covered = True
        
        i = 0
        # (2) Iterate until we reach a point in range vertically
        if not all_points_covered:
            while points_in_x_range[i][1] < vertical_lb:
                i += 1

        # (3) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
        current_sum = 0
        current_pos = (p[0], points_in_x_range[i][1])
        j = i
        while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
            current_sum += points_in_x_range[j][2]
            j += 1

        if current_sum > best_sum:
            best_sum = current_sum
            best_pos = current_pos

        # (4) Iterate through points, altering window as we go, until we use the current left-most point as bottom
        # TODO: Should be able to add a condition here to stop if the next point is above the vertical_ub
        while j < len(points_in_x_range) and (all_points_covered or points_in_x_range[i] != p):
            # Remove weight of previous bottom point
            current_sum -= points_in_x_range[i][2]
            # Shift i to next point and reset current pos
            i += 1
            current_pos = (p[0], points_in_x_range[i][1])
            # Shift j and add new points
            while j < len(points_in_x_range) and points_in_x_range[j][1] <= points_in_x_range[i][1] + rect_h:
                current_sum += points_in_x_range[j][2]
                j += 1
            # Check new window
            if current_sum > best_sum:
                best_sum = current_sum
                best_pos = current_pos
        
        # (5) Now, we want to move i to point p in case we terminated early via the j condition, and remove that point
        if not all_points_covered:
            while points_in_x_range[i] != p:
                i += 1
            del points_in_x_range[i]
        else:
            break

    return best_pos + (best_sum,)

# # NOTE: Rehash of sortedList version using an AVL tree instead. Should also be fast theoretically
# # NOTE: I think this version will fail if we have any points with identical y-values, so not really practical
# def maxrs_sweepline_5(points, rect_w, rect_h):
#     # Let num points := n
#     # Sort time = n*log(n)
#     # For each candidate: n
#     #   Generate points in x range and push on to heap: m*log(m) where m <<< n (probably)
#     #   Unheap to sort them: m*log(m)
#     #   For each point in x range: m
#     #       Generate points in y range: m
#     #   Reheap elements: m*log(m)
#     # Runtime = n*log(n) + n*(3*m*log(m) + m(m)) = n*log(n) + 3*n*m*log(m) + n*m^2 = O(n*m^2)
#     # Assuming small m, this could be very fast
#     """
#     Sweep line O(n² log n) version.
    
#     Key optimization: For each x-position, filter points ONCE,
#     then sweep through y-positions checking only filtered points.
    
#     Args:
#         points: list of (x, y, weight)
#         rect_w, rect_h: rectangle width and height
#     Returns:
#         (best_x, best_y, max_sum)
#     """
#     if not points:
#         return (0, 0, 0)
    
#     # Idea: Keep an AVLTree containing the points currently within range horizontally,
#     #       sorted by their values vertically. Then we don't need to filter for candidates 
#     #       each time. AVLTRee supports O(logn) insertion, O(logn) random access, somehow O(n)
#     #       traversal, and O(logn) deletion (although in worst case, which is rare, deletion
#     #       is O(nlogn))

#     # Points are tuples, which are sorted by first value by default
#     points_by_x_val = sorted(points)

#     best_sum = -np.inf
#     best_pos = None
    
#     # Create a SortedList to store the points in range, sorted by their y-values. Add first point
#     points_in_x_range = AVLTree()
#     points_in_x_range.insert(points_by_x_val[0][1], points_by_x_val[0]) # Sort by y-value
#     next_element_to_add = 1

#     # Once all points are within our x range, we can use each remaining as bottom of rectangle and break
#     all_points_covered = False

#     # Iterate from leftmost point
#     for p in points_by_x_val:
#         horizontal_ub = p[0] + rect_w
#         vertical_lb = p[1] - rect_h
#         vertical_ub = p[1] + rect_h

#         # (1) Add new points to sorted list
#         while next_element_to_add < len(points_by_x_val) and points_by_x_val[next_element_to_add][0] <= horizontal_ub:
#             points_in_x_range.insert(points_by_x_val[next_element_to_add][1], points_by_x_val[next_element_to_add])
#             next_element_to_add += 1

#         # (1a) MAYBE: Check if last point is now in range. If so, use every point as bottom and check, then break
#         if next_element_to_add == len(points_by_x_val):
#             all_points_covered = True
        
#         current_sum = 0
#         # (2) Iterate until we reach a point in range vertically
#         for p in points_in_x_range.values():
#             if not all_points_covered:
#                 if p[1] < vertical_lb:
#                     continue

#         # (3) Create sum for initial window. i serves as bottom (inclusive) and j as top (exclusive)
#         current_pos = (p[0], curr_points_in_x_range[i][1])
#         j = i
#         while j < len(curr_points_in_x_range) and curr_points_in_x_range[j][1] <= curr_points_in_x_range[i][1] + rect_h:
#             current_sum += curr_points_in_x_range[j][2]
#             j += 1

#         if current_sum > best_sum:
#             best_sum = current_sum
#             best_pos = current_pos

#         # (4) Iterate through points, altering window as we go, until we use the current left-most point as bottom
#         # TODO: Should be able to add a condition here to stop if the next point is above the vertical_ub
#         while j < len(curr_points_in_x_range) and (all_points_covered or curr_points_in_x_range[i] != p):
#             # Remove weight of previous bottom point
#             current_sum -= curr_points_in_x_range[i][2]
#             # Shift i to next point and reset current pos
#             i += 1
#             current_pos = (p[0], curr_points_in_x_range[i][1])
#             # Shift j and add new points
#             while j < len(curr_points_in_x_range) and curr_points_in_x_range[j][1] <= curr_points_in_x_range[i][1] + rect_h:
#                 current_sum += curr_points_in_x_range[j][2]
#                 j += 1
#             # Check new window
#             if current_sum > best_sum:
#                 best_sum = current_sum
#                 best_pos = current_pos
        
#         # (5) Now, we want to remove the leftmost point
#         if not all_points_covered:
#             points_in_x_range.remove(p[0])

#     return best_pos + (best_sum,)

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
np.random.seed(3)
n = 100
points = [(np.random.uniform(0, 50), 
           np.random.uniform(0, 50), 
           np.random.randint(1, 10)) 
          for _ in range(n)]

rect_w, rect_h = 5, 5

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

print("\n[3] SWEEP LINE Lance-heap VERSION")
start = time.time()
bx3, by3, bs3 = maxrs_sweepline_2(points, rect_w, rect_h)
time3 = time.time() - start
print(f"    Result: ({bx3:.2f}, {by3:.2f}) with sum = {bs3}")
print(f"    Time: {time3*1000:.4f} ms") # Convert to milliseconds

print("\n[4] SWEEP LINE Lance-improved VERSION")
start = time.time()
bx4, by4, bs4 = maxrs_sweepline_3(points, rect_w, rect_h)
time4 = time.time() - start
print(f"    Result: ({bx4:.2f}, {by4:.2f}) with sum = {bs4}")
print(f"    Time: {time4*1000:.4f} ms") # Convert to milliseconds

print("\n[5] SWEEP LINE Lance-sortedList VERSION")
start = time.time()
bx5, by5, bs5 = maxrs_sweepline_4(points, rect_w, rect_h)
time5 = time.time() - start
print(f"    Result: ({bx5:.2f}, {by5:.2f}) with sum = {bs5}")
print(f"    Time: {time5*1000:.4f} ms") # Convert to milliseconds

# print("\n[6] SWEEP LINE Lance-AVLTree VERSION")
# start = time.time()
# bx6, by6, bs6 = maxrs_sweepline_5(points, rect_w, rect_h)
# time6 = time.time() - start
# print(f"    Result: ({bx6:.2f}, {by6:.2f}) with sum = {bs6}")
# print(f"    Time: {time6*1000:.4f} ms") # Convert to milliseconds

# Verify results match using sum bs1 and bs2
print("\n" + "="*60)
if bs1 == bs2:
    print("✓ Both versions produce the same result!")
    if time1 > time2:
        print(f"✓ Sweep line is {time1/time2:.2f}x faster!")
else:
    print("✗ Results don't match - bug in implementation!")
print("="*60)
if bs1 == bs2 == bs3 == bs4 == bs5:
    print("✓ All versions produce the same result!")
    if time1 > time3:
        print(f"✓ Sweep line (Lance-heap) is {time1/time3:.2f}x faster than basic!")
    if time2 > time3:
        print(f"✓ Sweep line (Lance-heap) is {time2/time3:.2f}x faster than other sweepline!")
    else:
        print((f"✗ Other sweepline is {time3/time2:.2f}x faster than sweepline (Lance-heap)!"))

    if time1 > time4:
        print(f"✓ Sweep line (Lance-improved) is {time1/time4:.2f}x faster than basic!")
    if time2 > time4:
        print(f"✓ Sweep line (Lance-improved) is {time2/time4:.2f}x faster than other sweepline!")
    else:
        print((f"✗ Other sweepline is {time4/time2:.2f}x faster than sweepline (Lance-improved)!"))

    if time1 > time5:
        print(f"✓ Sweep line (Lance-sortedList) is {time1/time5:.2f}x faster than basic!")
    if time2 > time5:
        print(f"✓ Sweep line (Lance-sortedList) is {time2/time5:.2f}x faster than other sweepline!")
    else:
        print((f"✗ Other sweepline is {time5/time2:.2f}x faster than sweepline (Lance-sortedList)!"))

    # if time1 > time6:
    #     print(f"✓ Sweep line (Lance-AVLTree) is {time1/time6:.2f}x faster than basic!")
    # if time2 > time6:
    #     print(f"✓ Sweep line (Lance-AVLTree) is {time2/time6:.2f}x faster than other sweepline!")
    # else:
    #     print((f"✗ Other sweepline is {time6/time2:.2f}x faster than sweepline (Lance-AVLTree)!"))
else:
    print("✗ Results don't match - bug in implementation!")

# Visualization
fig, ax = plt.subplots(figsize=(10, 8))

ax.scatter([p[0] for p in points], [p[1] for p in points],
           s=[p[2]*30 for p in points], color='blue', alpha=0.6, 
           label='Points (size = weight)', zorder=3)

for i, (x, y, w) in enumerate(points):
    ax.annotate(f'{w}', (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.7, fontweight='bold')

# Show rectangles for both methods
rect = plt.Rectangle((bx1, by1), rect_w, rect_h,
                     edgecolor='red', facecolor='red', 
                     alpha=0.2, lw=3, label=f'Rectangle Op (sum={bs1})')
ax.add_patch(rect)

rect = plt.Rectangle((bx2, by2), rect_w, rect_h,
                     edgecolor='blue', facecolor='blue', 
                     alpha=0.2, lw=3, label=f'Rectangle sweep (sum={bs2})')
ax.add_patch(rect)

rect = plt.Rectangle((bx3, by3), rect_w, rect_h,
                     edgecolor='green', facecolor='green', 
                     alpha=0.2, lw=3, label=f'Rectangle sweep (Lance-heap) (sum={bs3})')
ax.add_patch(rect)

rect = plt.Rectangle((bx4, by4), rect_w, rect_h,
                     edgecolor='yellow', facecolor='yellow', 
                     alpha=0.2, lw=3, label=f'Rectangle sweep (Lance-improved) (sum={bs4})')
ax.add_patch(rect)

rect = plt.Rectangle((bx5, by5), rect_w, rect_h,
                     edgecolor='purple', facecolor='purple', 
                     alpha=0.2, lw=3, label=f'Rectangle sweep (Lance-sortedList) (sum={bs5})')
ax.add_patch(rect)

# rect = plt.Rectangle((bx6, by6), rect_w, rect_h,
#                      edgecolor='black',# facecolor='purple', 
#                      alpha=0.2, lw=3, label=f'Rectangle sweep (Lance-AVLTree) (sum={bs6})')
# ax.add_patch(rect)

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