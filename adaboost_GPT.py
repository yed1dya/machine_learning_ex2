import logging
import math
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


# create points from database: each point is p = (x, y, label)
# labels are 1, -1
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
    train, test = [], []
    n = len(points)
    indexes = random.sample(range(n), n // 2)
    for i, point in enumerate(points):
        if i in indexes:
            train.append(point)
        else:
            test.append(point)
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
def decide_h(rule, point) -> int:
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


# calculate the error of a rule over all weighted points
def rule_error(points: list, weights: list, rule) -> float:
    error = 0
    for i, (x, y, label) in enumerate(points):
        point = (x, y)
        predicted_label = decide_h(rule, point)
        if predicted_label != label:
            error += weights[i]
    return error


# Set up logging
logging.basicConfig(
    filename='adaboost_debug.log',
    level=logging.DEBUG,
    format='%(message)s',
    filemode='w'  # Overwrite log file on each run
)


# Modified adaboost_algo function with logging
# Added analysis of differences between the most important and second most important rules
def adaboost_algo(points: list, rules: list, iterations: int, k: int):
    n = len(points)
    point_weights = [1 / n] * n  # set all point weights to 1/n
    output_rules = []
    misclassified_tracker = [0] * n  # Track misclassification count

    for iteration in range(iterations):
        # find the best rule:
        best_rule, second_best_rule, min_error, second_min_error = None, None, float('inf'), float('inf')
        rule_errors = []  # For debugging

        for rule in rules:
            error = rule_error(points, point_weights, rule)
            rule_errors.append((rule, error))
            if error < min_error:
                second_best_rule, second_min_error = best_rule, min_error
                best_rule, min_error = rule, error
            elif error < second_min_error:
                second_best_rule, second_min_error = rule, error

        # calculate alpha for the best rule:
        alpha = 0.5 * math.log((1 - min_error) / (min_error + 1e-10))

        # Update weights:
        Z_t = 0  # find the sum of the updated weights:
        misclassified_points = []  # Track misclassified points
        for i in range(len(point_weights)):
            predicted_label = decide_h(best_rule, points[i])
            if predicted_label != points[i][2]:
                misclassified_points.append(i)
                misclassified_tracker[i] += 1
            point_weights[i] *= math.exp(-alpha * points[i][2] * predicted_label)
            Z_t += point_weights[i]
        point_weights = [p / Z_t for p in point_weights]  # normalize point weights
        output_rules.append((best_rule, alpha))

        # Analyze differences between the first and second best rules
        diff_points = []  # Points where first and second rules disagree
        for i, point in enumerate(points):
            label_first = decide_h(best_rule, point)
            label_second = decide_h(second_best_rule, point)
            if label_first != label_second:
                diff_points.append((i, point, label_first, label_second))

        # Logging information for all iterations
        logging.debug(f"Iteration {iteration + 1}:")
        logging.debug(f"Best rule: {best_rule}, Error: {min_error}, Alpha: {alpha}")
        logging.debug(f"Second best rule: {second_best_rule}, Error: {second_min_error}")
        logging.debug(f"Misclassified points by best rule: {misclassified_points}")
        logging.debug(f"Points where best and second best rules disagree: {diff_points}")
        logging.debug(f"Point weights: {point_weights}")
        logging.debug("\n")

    output_rules.sort(key=lambda x: x[1], reverse=True)  # sort the rules by alpha

    # Log consistently misclassified points
    logging.debug("Consistently Misclassified Points:")
    for i, count in enumerate(misclassified_tracker):
        if count > 0:
            logging.debug(f"Point {i}: Misclassified {count} times")

    return output_rules[:k]  # return the best k rules


# Visualize decision boundaries for selected rules
def visualize_decision_boundaries(rules, points, title):
    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.xlabel("X1")
    plt.ylabel("X2")

    # Plot points
    for x1, x2, label in points:
        color = 'red' if label == 1 else 'blue'
        plt.scatter(x1, x2, c=color, edgecolor='k')

    # Plot decision boundaries
    for (midpoint, slope, direction), alpha in rules:
        x_vals = np.linspace(min([p[0] for p in points]), max([p[0] for p in points]), 100)
        if slope == np.inf:  # Vertical line
            plt.axvline(x=midpoint[0], color='green', linestyle='--')
        else:
            y_vals = slope * x_vals + (midpoint[1] - slope * midpoint[0])
            plt.plot(x_vals, y_vals, linestyle='--', color='green', alpha=alpha / max([r[1] for r in rules]))

    plt.grid(True)
    plt.show()


# Quantify diversity of selected rules
def quantify_rule_diversity(rules, points):
    overlap_matrix = np.zeros((len(rules), len(rules)))
    for i, (rule1, _) in enumerate(rules):
        for j, (rule2, _) in enumerate(rules):
            if i < j:
                overlap = sum(
                    1 for point in points
                    if decide_h(rule1, point) == decide_h(rule2, point)
                ) / len(points)
                overlap_matrix[i, j] = overlap
                overlap_matrix[j, i] = overlap

    logging.debug("Rule Overlap Matrix:")
    logging.debug(overlap_matrix)
    return overlap_matrix


# final decision on a point, according to a set of weak rules
def H(rules, point):
    s = sum(alpha * decide_h(rule, point) for rule, alpha in rules)
    return 1 if s > 0 else -1


# calculate final rule errors
def calc_error(rules, points):
    return sum(1 for p in points if H(rules, (p[0], p[1])) != p[2]) / len(points)


# a single run of adaboost
def adaboost(points: list, rules: list, iterations: int, k: int):
    train, test = train_test(points)
    output_rules = adaboost_algo(train, rules, iterations, k)
    for rule in output_rules:
        print(f"rule: {rule[0]}, alpha: {rule[1]}")
    empirical_error = [calc_error(output_rules[:i], train) for i in range(1, k + 1)]
    true_error = [calc_error(output_rules[:i], test) for i in range(1, k + 1)]
    return empirical_error, true_error


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


# requirements for the assignment - how many runs, rules, which classes to use
def problem3(runs, seed, iterations, k, change):
    f = open(f'log_{TIMESTAMP}.txt', 'w')
    points = make_points_list('iris.txt', 'Iris-versicolor', 'Iris-virginica')
    if not points:
        return
    train, _ = train_test(points)
    rules = make_rules(train)

    # save errors from all runs
    empirical_errors, true_errors = [], []
    for i in range(runs):
        if change:
            random.seed(seed * i)  # update seed for each run
        print(f"run {i + 1}")
        e, t = adaboost(points, rules, iterations, k)
        empirical_errors.append(e)
        true_errors.append(t)
        # print(f"empirical errors:\n{e}")
        # print(f"true errors:\n{t}")
    f.close()

    # calculate average of each type of error
    avg_empirical = [round(float(np.mean([e[i] for e in empirical_errors])), 3) for i in range(iterations)]
    avg_true = [round(float(np.mean([t[i] for t in true_errors])), 3) for i in range(iterations)]

    print(f"Average Empirical Errors (Train Set, Averaged Across {runs} Runs):")
    print(avg_empirical)
    print(f"Average True Errors (Test Set, Averaged Across {runs} Runs):")
    print(avg_true)

    # save plots for analysis
    if change:
        title1 = f"empirical_changing-seed-{seed}_iterations-{iterations}_runs-{runs}_{TIMESTAMP}.png"
        title2 = f"true_changing-seed-{seed}_iterations-{iterations}_runs-{runs}_{TIMESTAMP}.png"
        plot1_title = (f"Average Empirical Errors, changing seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {TIMESTAMP}")
        plot2_title = (f"Average True Errors, changing seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {TIMESTAMP}")
    else:
        title1 = f"empirical_permanent-seed-{seed}_iterations-{iterations}_runs-{runs}_{TIMESTAMP}.png"
        title2 = f"true_permanent_seed-{seed}_iterations-{iterations}_runs-{runs}_{TIMESTAMP}.png"
        plot1_title = (f"Average Empirical Errors, permanent seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {TIMESTAMP}")
        plot2_title = (f"Average True Errors, permanent seed, {runs} runs,\n"
                       f"{iterations} iteration, timestamp: {TIMESTAMP}")

    save_plot(range(1, iterations + 1), avg_empirical, plot1_title, "Error Rate", title1)
    save_plot(range(1, iterations + 1), avg_true, plot2_title, "Error Rate", title2)


ITERATIONS = 8
HOW_MANY_RULES = 8
RUNS = 100

# set random seed for reproducibility
RANDOM_SEED = 9  # life, the universe, and everything
random.seed(RANDOM_SEED)
CHANGE_RANDOM_SEED = True


def main():
    print(f"Random seed set to {RANDOM_SEED} for reproducibility", end="")
    if CHANGE_RANDOM_SEED:
        print(f",\nand will increase by 1 each run.")
    else:
        print(".")
    problem3(RUNS, RANDOM_SEED, ITERATIONS, HOW_MANY_RULES, CHANGE_RANDOM_SEED)


if __name__ == '__main__':
    main()
