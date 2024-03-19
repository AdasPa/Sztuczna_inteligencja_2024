import numpy as np
import matplotlib.pyplot as plt

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy()
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy()

# DONE: calculate closed-form solution
X_train = np.column_stack((np.ones_like(x_train), x_train))
theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)

# DONE: calculate error
y_train_predicted = X_train.dot(theta_best)
mse_train = np.mean((y_train_predicted - y_train) ** 2)
print("MSE dla danych treningowych:", mse_train)


X_test = np.column_stack((np.ones_like(x_test), x_test))
y_test_predicted = X_test.dot(theta_best)
mse_test = np.mean((y_test_predicted - y_test) ** 2)
print("MSE dla danych testowych:", mse_test)


# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()

# TODO: standardization

x_train_mean = np.mean(x_train)
x_train_std = np.std(x_train)
x_train_standarized = (x_train - x_train_mean) / x_train_std

x_test_standarized = (x_test - x_train_mean) / x_train_std
X_train_standarized = np.column_stack((np.ones_like(x_train_standarized), x_train_standarized))

y_train_mean = np.mean(y_train)
y_train_std = np.std(y_train)
y_train_standarized = (y_train - y_train_mean) / y_train_std

y_test_standarized = (y_test - y_train_mean) / y_train_std


# TODO: calculate theta using Batch Gradient Descent
theta_best = [np.random.rand(), np.random.rand()]

rate = 0.01
iterations = 1000

for i in range(iterations):
    gradients = 2 / len(y_train) * X_train_standarized.T.dot(X_train_standarized.dot(theta_best) - y_train_standarized)
    theta_best -= rate*gradients

# TODO: calculate error
y_train_standarized_predicted = X_train_standarized.dot(theta_best)
mse_train_standarized = np.mean((y_train_standarized_predicted - y_train_standarized) ** 2)
print("MSE dla danych treningowych standaryzowanych G:", mse_train_standarized)


X_test_standarized = np.column_stack((np.ones_like(x_test_standarized), x_test_standarized))
y_test_standarized_predicted = X_test_standarized.dot(theta_best)
mse_test_standarized = np.mean((y_test_standarized_predicted - y_test_standarized) ** 2)
print("MSE dla danych testowych standaryzowanych G:", mse_test_standarized)


# plot the regression line
x = np.linspace(min(x_test_standarized), max(x_test_standarized), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test_standarized, y_test_standarized)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()