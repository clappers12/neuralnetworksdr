import numpy as np
import math
import warnings
import pandas
import matplotlib.pyplot as plt
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
warnings.simplefilter(action='ignore', category=FutureWarning)
file_path = r"/home/metapod/Desktop/take2/scientificProject/data.csv"
#if 'target_column' in data.columns:
#    data = data.drop('target_column', axis=1)
#try:
#    data = data.drop('target_column', axis=1)
#except KeyError:
#    print("'target_column' not found in the DataFrame.")

try:
    data = pandas.read_csv(file_path, sep=';', encoding='latin1')
except UnicodeDecodeError:
    try:
        data = pandas.read_csv(file_path, sep=';', encoding='ISO-8859-1')
    except UnicodeDecodeError:
        data = pandas.read_csv(file_path, sep=';', encoding='cp1252')
# Preparing the data for the RandomForestClassifier
X = data.drop('target_column', axis=1)
Y = data['target_column']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=24)

# Train the RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=24)
rf_classifier.fit(X_train, Y_train)
try:
    user_input = float(input("Enter a value for 'frequency' to predict: "))
    user_input_processed = np.array([[user_input]])  # Reshape input for prediction
    prediction = rf_classifier.predict(user_input_processed)
    print(f"Prediction (True/False): {prediction[0]}")
except ValueError:
    print("Invalid input. Please enter a numeric value.")

##
#file_path = r"/home/metapod/Desktop/take2/scientificProject/data.csv"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=24)

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=any)

# Train the classifier on the training data
rf_classifier.fit(X_train, Y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)
# y_pred = rf_classifier(10,0,2)
# Calculate the accuracy of the classifier
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
sparseMatrix = csr_matrix((3, 4), dtype=np.int8).toarray()
theta = 3.65 # pi over 180, once coupled with the other function.
phi = 1.618033988749895

# Create a 2D grid of theta and phi
theta_grid, phi_grid = np.meshgrid(theta, phi)


# Define the wave function Psi(theta, phi)
def wave_function(theta_grid):
    return np.sin(theta_grid) * np.cos(phi_grid)


#def Wave(Coordinates)



# Calculate the wave function values on the grid
wave_function_values = wave_function_var = (theta_grid, * phi_grid / phi)
# Define the range for x (avoiding zero and negative values)
x = np.linspace(0.1, 10, 400)

# Define a few values for phi
phi_values = [0, 0.5, 1, 1.5, 2]

# Plot the function for each phi
plt.figure(figsize=(12, 8))
for phi in phi_values:
    y = p.sin(np.log(x)) + phi
    plt.plot(x, y, label=f'phi = {phi}')

plt.title('Wave Function Iterations for sin(log(x)) + phi')
plt.xlabel('x')
plt.ylabel('sin(log(x)) + phi')
plt.legend()
plt.grid(True)
plt.show()
#wave_function_values = 356.10
vector = int(3.65)
theta_grid = np.array([16, 32, 64, 128, 256, 512])
phi_grid = np.array([1024, 2048, 4096, 8192, 16384, 32768])
#math.floor = 3.65 // math.floor(X)
print(int(3.13159 ** 5))
i = theta
result = (phi_grid + i)
#math.sinh(X)
x = 3.14159  # pi
sinh_result = math.sinh(x)
print(sparseMatrix)
try:
    user_input = float(input("Enter a value for 'compute' to predict: "))
    user_input_processed = np.array([[user_input]])  # Reshape input for prediction
    prediction = rf_classifier.predict(user_input_processed)
    print(f"Prediction (True/False): {prediction[0]}")
except ValueError:
    print("Invalid input. Please enter a numeric value.")