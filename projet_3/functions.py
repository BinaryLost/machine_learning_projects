import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.linear_model import LinearRegression


def showScatterPlot(data, x, y, ax=None):
    x_values = data[x].values
    y_values = data[y].values
    slope, intercept = np.polyfit(x_values, y_values, 1)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(x_values, y_values)
    ax.plot(x_values, slope * x_values + intercept, color='red')
    ax.set_xlabel(x)
    ax.set_ylabel(y)


def showHistogram(data, column, bins=50, ax=None):
    column_values = data[column].values
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(column_values, bins=bins, color='steelblue', density=True, edgecolor='none')
    ax.set_xlabel(column),
    ax.set_ylabel('Frequency')
    ax.set_title(column, fontsize=14)


def train_test_split_data(data, x, y):
    X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    return mse


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred, squared=False)
    return model, mse
