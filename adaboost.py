import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np


# plot the results for visualization
def save_plot(x_values, y_values, title, y_label, filename):
    plt.figure()
    plt.plot(x_values, y_values, marker='o')
    plt.title(title)
    plt.xlabel("Number of Rules")
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(filename)
    plt.close()


# parse data from 'iris.txt' to create the lists of points with labels
def make_points_list(file_name: str, class1: str, class2: str) -> list:
    try:
        points = []
        with open(file_name, 'r') as file:
            for line in file:
                t = line.split()
                if t[4] == class1:
                    points.append([float(t[1]), float(t[2]), 1])
                elif t[4] == class2:
                    points.append([float(t[1]), float(t[2]), -1])
        return points
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


# split points list into train and test (50% each)
# use random seed for reproducible randomness
def train_test(points: list) -> (list, list):
    indexes = random.sample(range(len(points)), len(points) // 2)
    train = [points[i] for i in indexes]
    test = [points[i] for i in range(len(points)) if i not in indexes]
    return train, test


def calculate_slope(p1, p2):
    if p1[0] == p2[0]:  # Vertical line
        return np.inf
    return (p2[1] - p1[1]) / (p2[0] - p1[0])


# create rules list
# each rule is defined by 2 points from the train set: p1, p2
# each rule is h = (point, slope)
# the point is the midpoint between p1 and p2
# the slope is the slope between p1 and p2
# each pair of points actually defines 2 rules -
# one side is positive and the other negative, and the opposite
def make_rules(points: list) -> list:
    rules = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points):
            if i < j:
                mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                slope = calculate_slope(p1, p2)
                rules.append(((mx, my), slope, 1))
                rules.append(((mx, my), slope, -1))
    return rules


# guess a label for a point according to a rule
def weak_h_decision(rule, point) -> int:
    mx, my, slope, direction = rule[0][0], rule[0][1], rule[1], rule[2]
    px, py = point[0], point[1]

    # find intercept of rule with y-axis:
    intercept = my - slope * mx if slope != np.inf else None

    # if line is vertical, decide by if point is to left or right:
    if slope == np.inf:
        return 1 if (px > mx and direction == 1) or (px < mx and direction == -1) else -1

    # else, decide by if point is above or below line:
    return 1 if (py > slope * px + intercept and direction == 1) or (
                py < slope * px + intercept and direction == -1) else -1


# final decision on a point, according to a set of weak rules
def strong_h_decision(rules, point):
    s = sum(alpha * weak_h_decision(rule, point) for rule, alpha in rules)
    return 1 if s > 0 else -1


# calculate final rule errors
def calc_error(rules, points):
    return sum(1 for p in points if strong_h_decision(rules, (p[0], p[1])) != p[2]) / len(points)


# calculate the error of a rule over all weighted points
def rule_error(points: list, weights: list, rule) -> float:
    error = 0
    for i, (x, y, label) in enumerate(points):
        point = (x, y)
        predicted_label = weak_h_decision(rule, point)
        if predicted_label != label:
            error += weights[i]
    return error


# adaboost algorithm steps
def adaboost_algo(points: list, rules: list, iterations: int, k: int):
    n = len(points)
    point_weights = [1 / n] * n  # set all point weights to 1/n
    output_rules = []

    for _ in range(iterations):
        # find the best rule:
        best_rule, min_error = None, float('inf')
        for rule in rules:
            error = rule_error(points, point_weights, rule)
            if error < min_error:
                min_error, best_rule = error, rule

        # calculate alpha for the best rule:
        alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))

        # update point weights:
        Z_t = 0  # find the sum of the updated weights:
        for i in range(len(point_weights)):
            point_weights[i] *= math.exp(-alpha * points[i][2] * weak_h_decision(best_rule, points[i]))
            Z_t += point_weights[i]
        point_weights = [p / Z_t for p in point_weights]  # normalize point weights
        output_rules.append((best_rule, alpha))

    output_rules.sort(key=lambda x: x[1], reverse=True)  # sort the rules by alpha
    return output_rules[:k]  # return the best k rules


# a single run of adaboost
def adaboost(train: list, test: list, rules: list, iterations: int, k: int):
    output_rules = adaboost_algo(train, rules, iterations, k)
    for rule in output_rules:
        print(f"rule: {rule[0]}, alpha: {rule[1]}")
    empirical_error = [calc_error(output_rules[:i], train) for i in range(1, k + 1)]
    true_error = [calc_error(output_rules[:i], test) for i in range(1, k + 1)]
    return empirical_error, true_error


# requirements for the assignment - how many runs, rules, which classes to use
def problem3(runs, seed, iterations, k, change):
    points = make_points_list('iris.txt', 'Iris-versicolor', 'Iris-virginica')
    if not points:
        return

    # save errors from all runs
    empirical_errors, true_errors = [], []
    for i in range(runs):
        train, test = train_test(points)
        rules = make_rules(train)
        if change:
            random.seed(seed + (i ** 2) - (i // 3))  # update seed for each run
        print(f"run {i + 1}")
        e, t = adaboost(train, test, rules, iterations, k)
        empirical_errors.append(e)
        true_errors.append(t)

    # calculate average of each type of error
    avg_empirical = [float(np.mean([e[i] for e in empirical_errors])) for i in range(iterations)]
    avg_true = [float(np.mean([t[i] for t in true_errors])) for i in range(iterations)]

    print(f"Average Empirical Errors (Train Set, Averaged Across {runs} Runs):")
    print(avg_empirical)
    print(f"Average True Errors (Test Set, Averaged Across {runs} Runs):")
    print(avg_true)

    # save plots for analysis
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    if change:
        title1 = f"empirical_changing-seed-{seed}_iterations-{iterations}_runs-{runs}_{timestamp}.png"
        title2 = f"true_changing-seed-{seed}_iterations-{iterations}_runs-{runs}_{timestamp}.png"
        plot1_title = (f"Average Empirical Errors, changing seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {timestamp}")
        plot2_title = (f"Average True Errors, changing seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {timestamp}")
    else:
        title1 = f"empirical_permanent-seed-{seed}_iterations-{iterations}_runs-{runs}_{timestamp}.png"
        title2 = f"true_permanent_seed-{seed}_iterations-{iterations}_runs-{runs}_{timestamp}.png"
        plot1_title = (f"Average Empirical Errors, permanent seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {timestamp}")
        plot2_title = (f"Average True Errors, permanent seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {timestamp}")

    save_plot(range(1, iterations + 1), avg_empirical, plot1_title, "Error Rate", title1)
    save_plot(range(1, iterations + 1), avg_true, plot2_title, "Error Rate", title2)


ITERATIONS = 20
HOW_MANY_RULES = 10
RUNS = 100

# set random seed for reproducibility
# the seed will change between runs
RANDOM_SEED = 42  # life, the universe, and everything
random.seed(RANDOM_SEED)
CHANGE_RANDOM_SEED = True


def main():
    print(f"Random seed set to {RANDOM_SEED} for reproducibility", end="")
    if CHANGE_RANDOM_SEED:
        print(f",\nand will change each run.")
    else:
        print(".")
    problem3(RUNS, RANDOM_SEED, ITERATIONS, HOW_MANY_RULES, CHANGE_RANDOM_SEED)


if __name__ == '__main__':
    main()
