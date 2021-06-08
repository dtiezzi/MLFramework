from . import funcs
from . import plotreport
import scikitplot as skplt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score
import datetime
import pickle

class Run:

    def __init__(self, model, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = funcs.labelencode(y_train, y_test)[0]
        self.y_test = funcs.labelencode(y_train, y_test)[1]
        self.model = model
        self.y_pred = None
        self.fittedModel = None

    def runModel(self, name, pjname, confusion_matrix=True, random_state=42):
        dt = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.fittedModel = self.model.fit(self.X_train, self.y_train)
        filename = './reports/models/' + pjname + "_" + name + "_" + dt + '.sav'
        pickle.dump(self.model, open(filename, 'wb'))
        self.y_pred = self.fittedModel.predict(self.X_test)
    
    def reportpred(self, pjname ,roc=True, normalizeCM=False, plot_calibration=True, title="", pos_label=1):
        name = title
        if hasattr(self.fittedModel, "predict_proba"):
            prob_pos = self.fittedModel.predict_proba(self.X_test)
        else:  # use decision function
            prob_pos = self.fittedModel.decision_function(self.X_test)
            prob_pos = (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
        sp = 1 if input("Save plots (y/n)?").lower() == 'y' else 0  
        plotreport.plotgraphs(self.y_test, self.y_pred, prob_pos, name, pjname, saveplots=sp)

    def loadmodel(self, filename):
        loaded_model = pickle.load(open(filename, 'rb'))
        result = loaded_model.score(self.X_test, self.y_test)
        print(result)

    def report(self, n_top=3):
        for i in range(1, n_top + 1):
            candidates = np.flatnonzero(self.fittedModel['rank_test_score'] == i)
            for candidate in candidates:
                print("Model with rank: {0}".format(i))
                print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    self.fittedModel['mean_test_score'][candidate],
                    self.fittedModel['std_test_score'][candidate]))
                print("Parameters: {0}".format(self.fittedModel['params'][candidate]))
                print("")
