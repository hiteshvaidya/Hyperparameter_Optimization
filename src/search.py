"""
Hyperparameter optimization using Grid and Random Search
"""

import pandas as pd
import numpy as np

from sklearn import ensemble
from sklearn import metrics
from sklearn import model_selection
from sklearn import decomposition
from sklearn import preprocessing
from sklearn import pipeline


def grid_search(X, y):
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    param_grid = {
        'n_estimators': [100, 200], #, 300, 400
        'max_depth': [1, 3, 5, 7],
        'criterion': ['gini', 'entropy']
    }

    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)


def random_search(X, y):
    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    rf = ensemble.RandomForestClassifier(n_jobs=-1)
    classifier = pipeline.Pipeline(
        [
            ('scaling', scl),
            ('pca', pca),
            ('rf', rf)
        ])
    param_grid = {
        'pca__n_components': np.arange(5, 10),
        'rf__n_estimators':np.arange(100, 1500, 100),
        'rf__max_depth': np.arange(1,20),
        'rf__criterion': ['gini', 'entropy']
    }

    model = model_selection.RandomizedSearchCV(
        estimator=classifier,
        param_distributions=param_grid,
        n_iter=10, #(n_iter only for randomsearch)
        scoring='accuracy',
        verbose=10,
        n_jobs=1,
        cv=5
    )
    model.fit(X, y)
    print(model.best_score_)
    print(model.best_estimator_.get_params())



if __name__ == '__main__':
    df = pd.read_csv('/home/group2/research/nac/data/hitesh/Hyperparameter Optimization/data/train.csv')
    X = df.drop('price_range', axis=1).values
    y = df.price_range.values

    grid_search(X, y)
    random_search(X, y)

