import matplotlib.pyplot as plt
from datetime import datetime


# code to plot the results visually
def plot_results(points, w, output_file):
    class1_points = [p for p in points if p[2] == 1]
    class2_points = [p for p in points if p[2] == -1]

    plt.figure(figsize=(8, 6))

    plt.scatter(
        [p[0] for p in class1_points],
        [p[1] for p in class1_points],
        color='blue', label='Class 1', s=10
    )
    plt.scatter(
        [p[0] for p in class2_points],
        [p[1] for p in class2_points],
        color='red', label='Class 2', s=10
    )

    if w[0] != 0 or w[1] != 0:  # Ensure w is not the zero vector
        x_vals = [min(p[0] - 1 for p in points), max(p[0] + 1 for p in points)]
        if w[1] != 0:  # Avoid division by zero
            slope = -w[0] / w[1]
            intercept = -w[2] / w[1]
            y_vals = [slope * x + intercept for x in x_vals]
        else:  # Vertical line
            x_vals = [-w[2] / w[0], -w[2] / w[0]]
            y_vals = [min(p[1] for p in points), max(p[1] for p in points)]
        plt.plot(x_vals, y_vals, color='green', label='Separating Line')

    plt.arrow(0, 0, w[0], w[1], color='purple', head_width=0.1, label='Weight Vector')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Perceptron Results')
    plt.legend()
    plt.grid(True)

    plt.axis('equal')

    plot_file = output_file.replace('.txt', '.png')
    plt.savefig(plot_file)
    plt.close()
    print(f"Plot saved to {plot_file}")


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
    except Exception as e:
        print(f"An error occurred: {e}")


# calculation of dot product of 2 vectors,
# ignoring the last index (the label)
def dot_product(x: list, y: list):
    if len(x) != len(y):
        return None
    m = len(x) - 1
    prod = 0
    for i in range(m):
        prod += (x[i] * y[i])
    return prod


# adding a point to the 'w' vector,
# ignoring the label.
# the label '1' is added because 'w' is always labeled '1'
def add_vectors(w: list, x: list):
    if len(w) != len(x):
        return None
    m = len(w) - 1
    v = []
    for i in range(m):
        v.append(w[i] + x[i])
    v.append(1)
    return v


# subtracting a point from the 'w' vector
def sub_vectors(x: list, y: list):
    if len(x) != len(y):
        return None
    m = len(x) - 1
    v = []
    for i in range(m):
        v.append(x[i] - y[i])
    v.append(1)
    return v


# perceptron algorithm steps
def perceptron_algo(points: list, iterations, output_file: str):
    f = open(output_file, 'w')
    w = [0, 0, 1]
    i, mistake_counter = 0, 0
    mistake = True
    while mistake and i < iterations:
        i += 1
        f.write(f"Iteration: {i}\n")
        mistake = False
        for x in points:
            prod = dot_product(w, x)
            guess = 1 if prod > 0 else -1
            f.write(f"w: {w}, x: {x}, dot product: {prod}, guess: {guess}\n")
            if guess != x[len(x) - 1]:
                if x[len(x) - 1] == 1:
                    w = add_vectors(w, x)
                else:
                    w = sub_vectors(w, x)
                mistake = True
                mistake_counter += 1
                break
    f.write(f"\nIterations: {i}, mistakes: {mistake_counter}, w: {w[:2]}")
    f.close()
    plot_results(points, w, output_file)
    return i, mistake_counter, w[:2]


# 'wrapper' for the algorithm; handles input and output
def perceptron(file_name: str, class1: str, class2: str, max_iterations=None):
    timestamp = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    points_list = make_points_list(file_name, class1, class2)
    output_file = f"{class1.split('-')[1]}-{class2.split('-')[1]}"
    output_file += f"_max-iter_{max_iterations}_{timestamp}.txt"
    ans = perceptron_algo(points_list, max_iterations, output_file)
    print(f"{class1}, {class2}: Iterations: {ans[0]}, mistakes: {ans[1]}, w: {ans[2]}")


def main():
    perceptron('iris.txt', 'Iris-setosa', 'Iris-versicolor', 200)
    perceptron('iris.txt', 'Iris-setosa', 'Iris-virginica', 200)
    perceptron('iris.txt', 'Iris-versicolor', 'Iris-virginica', 200)


if __name__ == "__main__":
    main()
