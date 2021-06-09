from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from . import models
from . import rfparams
import numpy as np
import warnings
warnings.filterwarnings('ignore')
#from multiprocessing import Process, Pool, cpu_count

warnings.filterwarnings("ignore")
rfpg = rfparams.Rfparams()

class Grid:

    def __init__(self):
        self.dt_model = DecisionTreeClassifier(random_state=42)
        self.dt_params_grid = {"criterion": ["gini", "entropy"],
                "min_samples_split": [2,5,8,15,20],
                "max_depth": [2,4,6,8,10],
                "min_samples_leaf": [1,2,4,8,10],
                "max_leaf_nodes": [2,4, 7,9, 12, 20],
                }
        self.svm_model = SVC(class_weight='balanced', probability=True)
        self.svm_params_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]
        
        self.rf_model = RandomForestClassifier(random_state=42, verbose=1)
        self.rf_params_grid = {'n_estimators': rfpg.n_estimators,
               'max_features': rfpg.max_features,
               'max_depth': rfpg.max_depth,
               'min_samples_split': rfpg.min_samples_split,
               'min_samples_leaf': rfpg.min_samples_leaf,
               'bootstrap': rfpg.bootstrap}
        self.lr_model = LogisticRegression()
        self.xgb_model = XGBClassifier(objective='binary:logistic', eval_metric="logloss" , nthread=16)
        self.xgb_params_grid = {
                'learning_rate': [0.02, 0.03, 0.04],
                'n_estimators': [500,700,800],
                'min_child_weight': [1, 5, 10],
                'gamma': [0.5, 1, 1.5, 2, 5],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0],
                'max_depth': [5,10,20,40]
                }
        # self.xgb_params_grid = {
        #         'learning_rate': [0.1, 0.05],
        #         'n_estimators': [500,700],
        #         'min_child_weight': [1, 5],
        #         'gamma': [0.5, 1],
        #         'max_depth': [5,30]
                # }
        self.ann_model = neural_network.MLPClassifier(verbose=True)
        self.ann_params_grid = {'activation':['relu', 'tanh'],
                'solver': ['adam', 'lbfgs', 'sgd'],
                'max_iter': [500,1000], 
                'alpha': 10.0 ** -np.arange(1, 3), 
                'hidden_layer_sizes':[2,3,5,7], 
                'batch_size':np.arange(5,15,32),
                'learning_rate_init':[0.01, 0.03, 0.1]
                }
        self.knn_model = KNeighborsClassifier()
        self.knn_params_grid = {
                'n_neighbors': np.arange(3,15,2),
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
                }
        self.md = {'dt' : [self.dt_model, self.dt_params_grid],
                    'svm' : [self.svm_model, self.svm_params_grid],
                    'rf' : [self.rf_model, self.rf_params_grid],
                    'xgb' : [self.xgb_model, self.xgb_params_grid],
                    'ann' : [self.ann_model, self.ann_params_grid], 
                    'knn' : [self.knn_model, self.knn_params_grid],
                    'lr' : [self.lr_model, None]}
    
    def report(results, n_top=3):
        for i in range(1, 3 + 1):
            candidates = np.flatnonzero(results['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results['mean_test_score'][candidate],
                    results['std_test_score'][candidate]))
                print("Parameters: {0}".format(results['params'][candidate]))
                print("")

    def runGrid(self, ops, X_train, y_train):
        m = models.Modelist()
        # n_cpu = cpu_count()
        # process_list = []
        for op in ops:
            if op == 'xgb':
                gs = GridSearchCV(self.md[op][0], param_grid=self.md[op][1], scoring="roc_auc", n_jobs=10, cv=10)
                gs.fit(X_train, y_train)
            elif op == 'ann':
                gs = GridSearchCV(self.md[op][0], param_grid=self.md[op][1], n_jobs=-1, scoring='roc_auc')
                gs.fit(X_train, y_train)
                gs.best_estimator_.__dict__['early_stopping'] = False
                gs.best_estimator_.__dict__['n_iter_no_change'] = 20
            elif op == 'lr':
                pass
            elif op == 'svm':
                gs = GridSearchCV(self.md[op][0], cv=3, param_grid=self.md[op][1], n_jobs=-1)
                gs.fit(X_train, y_train)
                gs.best_estimator_.__dict__['predict_proba'] = True
            elif op == 'knn':
                gs = GridSearchCV(self.md[op][0], param_grid=self.md[op][1], cv=3, n_jobs=-1)
                gs.fit(X_train, y_train)
            else:
                gs = GridSearchCV(self.md[op][0], cv = 3, param_grid=self.md[op][1], n_jobs=-1)
                gs.fit(X_train, y_train)
            
            m.modelist[op][1] = gs.best_estimator_ if op != 'lr' else self.lr_model

            
        return m
