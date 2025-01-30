DATA_DIR = "data"
DATA_FILE = "data.csv"
BEST_MODELS_DIR = "ML/Models"
RESULTS_DIR = "Results"
import os
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier , ExtraTreeClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import ElasticNet , RidgeClassifier , Lasso , LogisticRegression  , PassiveAggressiveClassifier
from catboost import CatBoostClassifier 
MAX_ITER = 1008
MODEL_DICT = {
    "gbc": GradientBoostingClassifier(),
    "xgbc": {#true implies binary classification else multi-class
        True: XGBClassifier(objective='binary:logitraw', n_estimators=100),
        False: XGBClassifier(objective='multi:softprob', n_estimators=100),
        },
    "dt": DecisionTreeClassifier(),
    "rf": RandomForestClassifier(),
    "svm": SVC(probability=True, max_iter=MAX_ITER),
    "knn": KNeighborsClassifier(),
    "enet": ElasticNet(max_iter=MAX_ITER),
    "ridge": RidgeClassifier(max_iter=MAX_ITER),
    "lasso": Lasso(max_iter=MAX_ITER),
    "logReg": LogisticRegression(max_iter=MAX_ITER),
    "ada": AdaBoostClassifier(),
    "pac": PassiveAggressiveClassifier(max_iter=MAX_ITER),
    "et": ExtraTreeClassifier(),
    "cat": CatBoostClassifier(),
    # "rbm": BernoulliRBM(n_components=256, learning_rate=0.01, n_iter=MAX_ITER)
}
 
MODEL_PARAMS = {
    "gbc": {'learning_rate': [0.1, 0.5, 1],
            'n_estimators': [100, 200, 300],
            'criterion': ['friedman_mse', 'squared_error'],
            'max_depth': [3, 5, 7, 10],
            },
    "xgbc": {'learning_rate': [0.1, 0.5, 1],
             'max_depth': [3, 5, 7, 10],
             'alpha': [0.1, 1, 10],
             'booster': ['gbtree', 'gblinear'],
             'eta': [0.01, 0.05, 0.07],
             'min_child_weight': [3, 5, 7, 9]
             },
    "dt": {'max_depth': [3, 5, 7, 10],
            'splitter': ['best', 'random'],
            'criterion': ['gini', 'entropy','log_loss'],
            'min_samples_split': [3, 5, 7, 10]
            },
    "et": {'max_depth': [3, 5, 7, 10],
                   'splitter': ['best', 'random']
                   },
    "svm": {'C': [0.1, 1, 10],
            'kernel': ['linear', 'poly','rbf','sigmoid'],
            'degree': [4, 6, 8, 10],
            'gamma': ['scale', 'auto',0.001]
            },
    "rf": {'n_estimators': [100, 200, 300],
           'criterion': ['gini', 'entropy', 'log_loss'],
           'min_samples_split': [3, 5, 7, 10],
           'max_depth': [3, 5, 7, 10],
           'max_features': ['sqrt', 'log2']
           },
    "ada": {'n_estimators': [100, 200, 300],
            'learning_rate': [0.1, 0.5, 1]
            },
    "knn": {'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [40,50,60]
            },
    "enet": {'alpha': [0.1, 1, 10],
             'l1_ratio': [0.01, 0.03, 0.05, 0.09],
             'fit_intercept': [True, False],
             'max_iter' : [1000, 2000, 3000],
             'selection': ['cyclic', 'random']
             },
    "ridge": {'alpha': [0.1, 1, 10],
              'solver' : ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag'],
              'fit_intercept': [True, False],
              'max_iter' : [1000, 2000, 3000]
              },
    "lasso": {'alpha': [0.1, 1, 10],
              'selection': ['cyclic', 'random'],
              'fit_intercept': [True, False],
              'max_iter' : [1000, 2000, 3000]
              },
    "logReg": {'C': [0.1, 1, 10],
               'penalty' : ['l1', 'l2', 'elasticnet', None],
               'dual' : [True, False],
               'fit_intercept': [True, False],
               'multi_class': ['auto', 'ovr', 'multinomial']
               },
    "pac":{'C': [0.1, 1, 10]},
    "cat": {
            'depth': [3, 5, 7, 9],                 
            'iterations': [50, 100, 200, 500],    
            'learning_rate': [0.01, 0.05, 0.1, 0.5]
            },

}