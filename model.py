import numpy as np
import matplotlib as plt
def wave_function(x, phi):
    return np.sin(np.log(x)) + phi

# Define the range for x and phi
x = np.linspace(0.1, 10, 400)
phi_values = [0, 0.5, 1, 1.5, 2]

# Plot the function for each phi
plt.figure(figsize=(12, 8))
for phi in phi_values:
    y = wave_function(x, phi)
    plt.plot(x, y, label=f'phi = {phi}')

plt.title('Wave Function: sin(log(x)) + phi')
plt.xlabel('x')
plt.ylabel('Wave Function Value')
plt.legend()
plt.grid(True)
plt.show()
# Calculate the wave function values on the grid
#wave_function_values = wave_function(theta_grid, phi_grid)
# Define the grid size and create arrays for theta and phi
grid_size = 64
theta = np.linspace(0, np.pi, grid_size)
phi = np.linspace(0, 2 * np.pi, grid_size)
theta_grid, phi_grid = np.meshgrid(theta, phi)

# Define the wave function
def wave_function(x, phi):
    return np.sin(np.log(x)) + phi

# Apply the wave function in the context of the grid
x_values = np.linspace(0.1, 10, grid_size)
wave_function_values = wave_function(x_values[:, np.newaxis], phi_grid)

# Plotting (example plot, adjust as needed)
plt.figure(figsize=(12, 8))
plt.contourf(theta_grid, phi_grid, wave_function_values, cmap='viridis')
plt.title('Wave Function in Polar Coordinates')
plt.xlabel('Theta')
plt.ylabel('Phi')
plt.colorbar(label='Wave Function Value')
plt.show()