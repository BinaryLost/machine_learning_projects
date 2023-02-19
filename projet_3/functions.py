import matplotlib.pyplot as plt
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV


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


def decision_tree(X_train, X_test, y_train, y_test, max_depth=3):
    tree_reg = DecisionTreeRegressor(max_depth=max_depth)
    tree_reg.fit(X_train, y_train)
    y_pred = tree_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return tree_reg, mse


def show_grid_analysis(X_train, y_train, max_depth):
    param_grid = {'max_depth': max_depth}
    tree_reg = DecisionTreeRegressor()
    grid_search = GridSearchCV(tree_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    results = grid_search.cv_results_
    df = pd.DataFrame(results)
    best_depth = grid_search.best_params_['max_depth']
    return best_depth, df[['params', 'mean_test_score', 'std_test_score']]


def evaluate_model(X_train, X_test, y_train, y_test, best_depth):
    tree_reg = DecisionTreeRegressor(max_depth=best_depth)
    tree_reg.fit(X_train, y_train)
    y_pred = tree_reg.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse
