import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor
from sklearn.svm import SVR


class CatWalk:
    """
    If you know what I mean...
    If you don't know what I mean: https://www.youtube.com/watch?v=P5mtclwloEQ
    Hint: what do models do? They walk the catwalk.
    """
    
    # Define attributes
    def __init__(self, df):
        self.X = df.drop(columns=TARGET)
        self.y = df[TARGET] 
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, 
            self.y, 
            random_state=RANDOM_STATE, 
            train_size=0.60
        )
        # The fitted models will be stored in a dicionary 
        # e.g. {'name', (bool, Model)}
        # The key is the name of the model we supplied 
        # The value is a tupble (bool, Model)
        ## Bool == True -> GridSerach; Bool == False -> Non GridSerach
        ## Model: is the fitted model itself
        self._fitted_models = {} 
    
    # Define a private method to accumulate fitted models
    def _save_model(self, name, model, is_grid_search=False):
        self._fitted_models[name] = (is_grid_search, model)
    
    # Define a method to fit a Non-GridSerach model
    def fit_model(self, name, fitter):
        model = fitter.fit(self.X_train, self.y_train)
        self._save_model(name, model)
        return self
    
    # Define a method to fit a GridSerach model    
    def fit_grid_search(self, name, pipe=None, params=None):
        pipe = Pipeline(steps=[]) if pipe is None else pipe
        params = {} if params is None else params
        
        model = GridSearchCV(
            pipe, 
            params, 
            cv=CV
        ).fit(self.X_train, self.y_train)
        
        self._save_model(name, model, is_grid_search=True)
        return self
    
    # Define a method to print train, test, and cross_val scores
    def print_results(self, name):
        grid_search, model = self._fitted_models[name]
        if not grid_search:
            print(f'train score: {model.score(self.X_train, self.y_train)}')
            print(f'test score: {model.score(self.X_test, self.y_test)}')
            print(f'cv score: {cross_val_score(model, self.X, self.y, scoring=METRIC, cv=CV).mean()}')
        else: 
            print(f'best params: {model.best_params_}')
            print(f'train score: {model.score(self.X_train, self.y_train)}')
            print(f'test score: {model.score(self.X_test, self.y_test)}')
            print(f'cv score: {cross_val_score(model.best_estimator_, self.X, self.y, scoring=METRIC, cv=CV).mean()}')      

    # Method to fit a non-pipeline or gridserach model and print result all in one motion
    def do_the_simple_turn(self, name, fitter):
        self.fit_model(name, fitter)
        self.print_results(name)
        return self
    
    # Method to fit a gridserach model and print result all in one motion
    def do_the_little_turn(self, name, pipe, params):
        self.fit_grid_search(name, pipe, params)
        self.print_results(name)
        return self
    
    # Method to return a list of model names
    def model_name_list(self):
        return list(self._fitted_models.keys())
    
    # Method to return the fitted model itself
    def get_model(self, name):
        return self._fitted_models[name][1]
    
    # Method to print all test scores
    def view_all_test_scores(self):
        for name, value in self._fitted_models.items():
            grid_search, model = value
            if grid_search:
                print(f'{name} test score: {model.score(self.X_test, self.y_test)}')
            else: 
                print(f'{name} test score: {model.score(self.X_test, self.y_test)}')
    
    # Method to make predictions
    def make_predictions(self, name, input_data):
        preds = self.get_model(name).predict(input_data)
        return preds