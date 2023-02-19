import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression


def showScatterPlot(data, x, y):
    x_values = data[x].values
    y_values = data[y].values
    slope, intercept = np.polyfit(x_values, y_values, 1)
    plt.scatter(x_values, y_values)
    plt.plot(x_values, slope * x_values + intercept, color='red')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()


def dummy_regressor_model(data, x, y):
    X = data[x]
    Y = data[y]
    model = DummyRegressor(strategy='mean')
    model.fit(X, Y)
    mean_pred = model.predict(X)
    print("Prédiction moyenne : ", mean_pred)


def train_test_split_data(data, x, y):
    X_train, X_test, y_train, y_test = train_test_split(data[x], data[y], test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


def get_grid(model, X_train, y_train, param_grid):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    results = grid_search.cv_results_
    df = pd.DataFrame(results)
    return grid_search, df


def evaluate_model(model,X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return model, mse

