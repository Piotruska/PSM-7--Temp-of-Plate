import numpy as np
import matplotlib.pyplot as plt


def initialize_temperatures(grid, top_temp, left_temp, bottom_temp, right_temp):
    """Assigns the initial boundary temperatures to the edges of the grid."""
    grid[0, :] = top_temp
    grid[-1, :] = bottom_temp
    grid[:, 0] = left_temp
    grid[:, -1] = right_temp


def construct_system_matrix(size):
    """Constructs the system matrix A for the temperature calculation and the vector b."""
    num_vars = (size - 2) ** 2
    A = np.zeros((num_vars, num_vars))
    b = np.zeros(num_vars)
    index = lambda i, j: (i - 1) * (size - 2) + (j - 1)

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
    return A, b


def apply_boundary_conditions_to_vector(b, grid):
    """Applies the boundary conditions to the vector b based on the boundary temperatures in grid."""
    size = grid.shape[0]
    index = lambda i, j: (i - 1) * (size - 2) + (j - 1)

    for i in range(1, size - 1):
        b[index(i, 1)] -= grid[i, 0]  # Left
        b[index(i, size - 2)] -= grid[i, -1]  # Right
    for j in range(1, size - 1):
        b[index(1, j)] -= grid[0, j]  # Top
        b[index(size - 2, j)] -= grid[-1, j]  # Bottom


def solve_linear_system(A, b):
    """Solves the linear system Ax = b using a numerical solver."""
    return np.linalg.solve(A, b)


def update_temperature_grid(grid, temperatures):
    """Updates the grid with the calculated internal temperatures."""
    dim = grid.shape[0]
    temp_reshaped = temperatures.reshape((dim - 2, dim - 2))
    grid[1:-1, 1:-1] = temp_reshaped


def display_temperature_grid(grid):
    """Displays the temperature grid as a heatmap."""
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='hot', origin='upper',interpolation='bilinear')
    plt.colorbar(label='Temperature (°C)')
    plt.title('Temperature Distribution on the Plate')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.show()


def save_grid_to_csv(grid, filename='temperature_distribution.csv'):
    """Saves the temperature grid to a CSV file."""
    np.savetxt(filename, grid, delimiter=',', fmt='%0.2f')


def main():
    """Main function to setup and solve the temperature distribution problem."""
    size = 41
    temperatures = [200, 100, 150, 50]  # Top, Left, Bottom, Right temperatures
    grid = np.zeros((size, size))

    initialize_temperatures(grid, *temperatures)
    A, b = construct_system_matrix(size)
    apply_boundary_conditions_to_vector(b, grid)
    solution = solve_linear_system(A, b)
    update_temperature_grid(grid, solution)

    display_temperature_grid(grid)
    save_grid_to_csv(grid)


if __name__ == "__main__":
    main()
