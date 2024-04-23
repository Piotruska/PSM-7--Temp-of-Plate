import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def initialize_temperatures(grid, top_temp, left_temp, bottom_temp, right_temp):
    grid[0, :] = top_temp
    grid[-1, :] = bottom_temp
    grid[:, 0] = left_temp
    grid[:, -1] = right_temp

def construct_system_matrix(size):
    num_vars = (size - 2) ** 2
    A = lil_matrix((num_vars, num_vars))
    b = np.zeros(num_vars)

    def index(i, j):
        return (i - 1) * (size - 2) + (j - 1)

    for i in range(1, size - 1):
        for j in range(1, size - 1):
            ind = index(i, j)
            A[ind, ind] = -4
            if i > 1:
                A[ind, index(i - 1, j)] = 1
            if i < size - 2:
                A[ind, index(i + 1, j)] = 1
            if j > 1:
                A[ind, index(i, j - 1)] = 1
            if j < size - 2:
                A[ind, index(i, j + 1)] = 1
    return A.tocsr(), b

def apply_boundary_conditions_to_vector(b, grid):
    size = grid.shape[0]
    index = lambda i, j: (i - 1) * (size - 2) + (j - 1)

    for i in range(1, size - 1):
        b[index(i, 1)] -= grid[i, 0]
        b[index(i, size - 2)] -= grid[i, -1]
    for j in range(1, size - 1):
        b[index(1, j)] -= grid[0, j]
        b[index(size - 2, j)] -= grid[-1, j]

def update_temperature_grid(grid, temperatures):
    dim = grid.shape[0]
    temp_reshaped = temperatures.reshape((dim - 2, dim - 2))
    grid[1:-1, 1:-1] = temp_reshaped

def display_temperature_grid(grid):
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='hot', origin='upper', interpolation='bilinear')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution on the Plate')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()

def main():
    size = 40
    temperatures = [200, 100, 150, 50]
    grid = np.zeros((size, size))

    initialize_temperatures(grid, *temperatures)
    A, b = construct_system_matrix(size)
    apply_boundary_conditions_to_vector(b, grid)
    solution = spsolve(A, b)
    update_temperature_grid(grid, solution)
    display_temperature_grid(grid)

if __name__ == "__main__":
    main()
