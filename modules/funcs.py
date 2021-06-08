from sklearn.preprocessing import LabelEncoder
import pandas as pd
from dtreeviz.trees import dtreeviz
import numpy as np
import os

def pjname():
    return input('Type the name of the project: ')

def dsetop():
    return input('Use the Breast Cancer dataset as example (Y/N)? ').lower()

def printFiles(f):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('#' * 67, '\n')
    print('We found the followin CSV files in your archive...')
    for n, i in enumerate(f):
        print('''
    {0} - {1}
        '''.format(n+1, i))
    return int(input('SELECT the file for machine learning analysis...')) - 1

def smtop():
    return 1 if input('Use SMOTE as a resampling method (Y/N)? ').lower() == 'y' else 0

def subset(df, op='y'):
    while op == 'y':
        cols = ['NA']
        cols = cols + list(df.columns)
        for n, i in enumerate(cols):
            print('''
        {0} - {1}    
        '''.format(n, i))
        c = int(input('Select a column for subset:'))
        os.system('cls' if os.name == 'nt' else 'clear')
        if c:
            col = cols[c]
            dt = df[col].dtype
            if dt == 'object':
                var = input('Type the variable to exclude:')
                df = df[df[col] != var]
            else:
                cutoff = float(input('Type the cut-off value:'))
                df = df[df[col] <= cutoff]
        op = input('Select another column (y/n)?').lower()
    return df

def printdf(df, msg):
    os.system('cls' if os.name == 'nt' else 'clear')
    print('\n' + '#' * 67 + '\n')
    print(msg)
    print(df.describe(include='all'))
    print('\n' + '#' * 67 + '\n')
    input('PRESS [ENTER] to Continue...')

def printoptions(ops, inst, multiple=0):
    os.system('cls' if os.name == 'nt' else 'clear')
    ops = list(ops)
    if multiple:
        ops.append('NA')
    print('''
    {0}
'''.format(inst))
    for i, o in enumerate(ops):
        print('''
    {0} - {1}
'''.format(i+1, o), end='')
    return ops[int(input())-1] if not multiple else [ops[i-1] for i in list(map(int, input().split()))]


def selectnumeric(df):
    return list(df.select_dtypes(include=['float64']).columns) + list(df.select_dtypes(include=['int64']).columns)

def selectfactors(df):
    return list(df.select_dtypes(include=['object']).columns)

def selectbinary(df, cols):
    return {key:[df[key].unique()] for key in cols}

def labelencode(y_train, y_test):
    le = LabelEncoder()
    le.fit(y_train)
    return [le.transform(y_train), le.transform(y_test)]

def printMloptions():
    os.system('cls' if os.name == 'nt' else 'clear')
    ops = {'dt' : 'Decision Tree', 'knn' : 'K-hearest Neighbor', 
            'svm' : 'Support Vector Machine', 'rf' : 'Random Forest', 
            'xgb' : 'XGBoost', 'lr' :  'Logistic Regression', 
            'ann' : 'Artificial Neural Network', 'all' : 'Apply all'}
    print('Select the following models:')
    for k in ops:
        print('''
    {0} - {1}
    '''.format(k, ops[k])
        )
    op = list(input().split())
    if 'all' in op:
        allMl = [k for k in ops]
        allMl.remove('all')
        return allMl
    else:
        return op
    
def plotTree(dt, X_train, X_test, y_train, target, class_nm):
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    y_train = label_encoder.transform(y_train)
    viz = dtreeviz(dt, X_train, y_train, target_name=target, feature_names=X_train.columns, class_names=list(class_nm)) #, orientation= 'LR'
    return viz