import matplotlib.pyplot as plt
import itertools

# Load the data from the file
data_file = 'iris.txt'

# Feature map
feature_map = {0: 'Sepal Length', 1: 'Sepal Width', 2: 'Petal Length', 3: 'Petal Width'}
feature1_idx = 1
feature2_idx = 2
# Iterate through all combinations of features
"""for feature1_idx, feature2_idx in itertools.combinations(range(4), 2):
    setosa_x, setosa_y = [], []
    versicolor_x, versicolor_y = [], []
    virginica_x, virginica_y = [], []

    with open(data_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 5:
                continue  # Skip malformed lines
            feature1, feature2, species = float(parts[feature1_idx]), float(parts[feature2_idx]), parts[4]
            if species == 'Iris-setosa':
                setosa_x.append(feature1)
                setosa_y.append(feature2)
            elif species == 'Iris-versicolor':
                versicolor_x.append(feature1)
                versicolor_y.append(feature2)
            elif species == 'Iris-virginica':
                virginica_x.append(feature1)
                virginica_y.append(feature2)

    # Plot the points
    # plt.scatter(setosa_x, setosa_y, color='orange', label='Setosa')
    plt.scatter(versicolor_x, versicolor_y, color='blue', label='Versicolor')
    plt.scatter(virginica_x, virginica_y, color='purple', label='Virginica')
    plt.grid(True)

    # Ensure equal axis lengths
    plt.axis('equal')

    # Add labels and legend
    plt.xlabel(feature_map[feature1_idx])
    plt.ylabel(feature_map[feature2_idx])
    plt.title(f'Iris Dataset: {feature_map[feature1_idx]} (Feature {feature1_idx}) vs {feature_map[feature2_idx]} (Feature {feature2_idx})')
    plt.legend()

    # Save the plot
    output_name = (f"{feature_map[feature1_idx].replace(' ', '_')}-{feature1_idx}"
                   f"_{feature_map[feature2_idx].replace(' ', '_')}-{feature2_idx}"
                   f"_iris.png")
    plt.savefig(output_name)
    plt.clf()"""


setosa_x, setosa_y = [], []
versicolor_x, versicolor_y = [], []
virginica_x, virginica_y = [], []

with open(data_file, 'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) != 5:
            continue  # Skip malformed lines
        feature1, feature2, species = float(parts[feature1_idx]), float(parts[feature2_idx]), parts[4]
        if species == 'Iris-setosa':
            setosa_x.append(feature1)
            setosa_y.append(feature2)
        elif species == 'Iris-versicolor':
            versicolor_x.append(feature1)
            versicolor_y.append(feature2)
        elif species == 'Iris-virginica':
            virginica_x.append(feature1)
            virginica_y.append(feature2)

# Plot the points
# plt.scatter(setosa_x, setosa_y, color='orange', label='Setosa')
plt.scatter(versicolor_x, versicolor_y, color='blue', label='Versicolor')
plt.scatter(virginica_x, virginica_y, color='purple', label='Virginica')
plt.grid(True)

# Ensure equal axis lengths
plt.axis('equal')

# Add labels and legend
plt.xlabel(feature_map[feature1_idx])
plt.ylabel(feature_map[feature2_idx])
plt.title(f'Iris Dataset: {feature_map[feature1_idx]} (Feature {feature1_idx}) vs {feature_map[feature2_idx]} (Feature {feature2_idx})')
plt.legend()

# Save the plot
output_name = (f"sv {feature_map[feature1_idx].replace(' ', '_')}-{feature1_idx}"
               f"_{feature_map[feature2_idx].replace(' ', '_')}-{feature2_idx}"
               f"_iris.png")
plt.savefig(output_name)
plt.clf()
