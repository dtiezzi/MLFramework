import pandas as pd
import numpy as np
from . import funcs
from sklearn.preprocessing import scale
from . import dfimputer
from sklearn import datasets

class ReadDb:

    def __init__(self):
        self.df = None
        self.X = None
        self.y = None
        self.target = None
        self.class_name = None

    def readfiles(self, file):
        self.df = pd.read_csv(file)

    def readbrca(self):
        brca = datasets.load_breast_cancer()
        self.df = pd.DataFrame(brca.data, columns=brca.feature_names)
        self.df['diagnostic'] = brca.target

    def preprocess(self):
        funcs.printdf(self.df, 'Database Summary')
        self.df = funcs.subset(self.df)
        y_op = funcs.printoptions(self.df.columns, "Select the dependent variable:")
        self.target = y_op
        y = self.df[y_op]
        self.class_name = y.unique()
        X = self.df.drop(y_op, axis=1)
        X_op = funcs.printoptions(X.columns, "Select the independent variable(s) to drop:", 1)

        for op in X_op:
            X = X.drop(op, axis=1) if op != 'NA' else X
        scaleCols = funcs.selectnumeric(X)
        for s in scaleCols:
            X[s] = scale(X[s])
        factorsCols = funcs.selectfactors(X)
        binaries = funcs.selectbinary(X, factorsCols)
        for k in binaries:
            if len(binaries[k][0]) <= 2:
                X[k] = X.loc[:, k].apply(lambda x: 0 if x == binaries[k][0][0] else 1).astype(int)
            else:
                X = pd.get_dummies(X, prefix=k, columns=[k])
        self.X = X
        self.y = y
        imp_op = input('Impute missing data (Y/N)?').lower()
        if imp_op == 'y':
            imp = dfimputer.Impute(self.X)
            imp.inputate()
            self.X = imp.df