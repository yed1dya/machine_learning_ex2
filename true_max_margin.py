import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from datetime import datetime


def calculate_slope(p1, p2):
    if p1[0] == p2[0]:  # vertical line
        return np.inf
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


# calculate the perpendicular distance from a point to a line
def point_line_distance(point, line):
    (m_x, m_y), slope = line
    if slope == np.inf:  # vertical line
        return abs(point[0] - m_x)
    intercept = m_y - slope * m_x
    return abs(slope * point[0] - point[1] + intercept) / np.sqrt(slope ** 2 + 1)


# check if all points on each side of the line have the same label
def is_valid_separator(line, points):
    (m_x, m_y), slope = line
    if slope == np.inf:
        left_labels = [p[2] for p in points if p[0] < m_x]
        right_labels = [p[2] for p in points if p[0] > m_x]
    else:
        intercept = m_y - slope * m_x
        left_labels = [p[2] for p in points if p[1] > slope * p[0] + intercept]
        right_labels = [p[2] for p in points if p[1] < slope * p[0] + intercept]
    return len(set(left_labels)) <= 1 and len(set(right_labels)) <= 1


# parse points from a file
def parse_points_from_file(filepath, class1, class2):
    points = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            x, y, label = float(parts[1]), float(parts[2]), parts[4]
            if label == class1:
                points.append([x, y, 1])
            elif label == class2:
                points.append([x, y, -1])
    return points


# find the line with the largest margin
def find_largest_margin(points):
    lines = []
    for i, j in combinations(points, 2):
        midpoint = [(i[0] + j[0]) / 2, (i[1] + j[1]) / 2]
        if i[0] == j[0]:
            slope = 0
        elif i[1] == j[1]:
            slope = np.inf
        else:
            slope = 1 / calculate_slope(i, j)
        lines.append((midpoint, slope))
        for k in points:
            if k[2] == i[2]:
                slope = calculate_slope(i, k)
                lines.append((midpoint, slope))
            elif k[2] == j[2]:
                slope = calculate_slope(j, k)
                lines.append((midpoint, slope))

    valid_lines = []
    for line in lines:
        if is_valid_separator(line, points):
            margins = [point_line_distance(p, line) for p in points]
            min_margin = min(margins)
            valid_lines.append((line, min_margin))

    if not valid_lines:
        # print("No valid lines found.")
        return None

    best_line = max(valid_lines, key=lambda x: x[1])
    print(f"Best line found: {best_line}, margin: {float(best_line[1])}")
    return best_line


# plot the results
def plot_points_and_line(points, line, class1, class2):
    plt.figure(figsize=(8, 8))
    for p in points:
        color = 'red' if p[2] == 1 else 'blue'
        plt.scatter(p[0], p[1], color=color)

    (m_x, m_y), slope = line[0]
    if slope == np.inf:  # Vertical line
        plt.axvline(x=m_x, color='green', linestyle='--')
    else:
        x_vals = np.linspace(min(p[0] for p in points) - 1, max(p[0] for p in points) + 1, 100)
        y_vals = slope * x_vals + (m_y - slope * m_x)
        plt.plot(x_vals, y_vals, color='green', linestyle='--')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Points and Line with Largest Margin')
    timestamp = datetime.now().strftime('%y-%m-%d_%H-%M-%S')
    filename = f"max_marg_{class1}_{class2}_{timestamp}.png"
    plt.savefig(filename)
    print(f"Plot saved as {filename}")


filepath = 'iris.txt'
class1 = 'Iris-virginica'
class2 = 'Iris-setosa'
points = parse_points_from_file(filepath, class1, class2)
result = find_largest_margin(points)
if result:
    print("Line with largest margin:", result)
    plot_points_and_line(points, result, class1, class2)
else:
    print("No valid separating line found.")
