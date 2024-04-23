import numpy as np
import matplotlib.pyplot as plt


def initialize_temperatures(size, top, left, bottom, right):
    matrix = np.zeros((size, size))
    matrix[0, :] = top
    matrix[-1, :] = bottom
    matrix[:, 0] = left
    matrix[:, -1] = right
    return matrix


def setup_matrix_and_vector(size):
    A = np.zeros(((size - 2) ** 2, (size - 2) ** 2))
    b = np.zeros((size - 2) ** 2)
    idx = lambda x, y: (x - 1) * (size - 2) + (y - 1)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            index = idx(i, j)
            A[index, index] = -4
            if i > 1:
                A[index, idx(i - 1, j)] = 1
            if i < size - 2:
                A[index, idx(i + 1, j)] = 1
            if j > 1:
                A[index, idx(i, j - 1)] = 1
            if j < size - 2:
                A[index, idx(i, j + 1)] = 1
    return A, b


def solve_temperature_distribution():
    size = 41
    top, left, bottom, right = 200, 100, 150, 50
    matrix = initialize_temperatures(size, top, left, bottom, right)
    A, b = setup_matrix_and_vector(size)

    for i in range(1, size - 1):
        b[(i - 1) * (size - 2)] -= matrix[i, 0]  # Left boundary
        b[(i - 1) * (size - 2) + size - 3] -= matrix[i, -1]  # Right boundary
    for j in range(1, size - 1):
        b[j - 1] -= matrix[0, j]  # Top boundary
        b[-j] -= matrix[-1, j]  # Bottom boundary

    A_inv = np.linalg.inv(A)
    T_internal = np.dot(A_inv, b)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            matrix[i, j] = T_internal[(i - 1) * (size - 2) + (j - 1)]

    return matrix


def plot_temperature_distribution(matrix):
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, cmap='hot', origin='upper')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution Across the Plate')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True, which='both', linestyle='-', color='w', linewidth=0.5)
    plt.xticks(np.arange(0, 41, 5))
    plt.yticks(np.arange(0, 41, 5))
    plt.show()


def write_to_csv(matrix, filename="result.csv"):
    np.savetxt(filename, matrix, delimiter=",", fmt='%0.3f')


if __name__ == "__main__":
    matrix = solve_temperature_distribution()
    plot_temperature_distribution(matrix)
    write_to_csv(matrix)
