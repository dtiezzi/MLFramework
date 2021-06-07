from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import neural_network
from xgboost import XGBClassifier
from sklearn.svm import SVC

class Modelist:

    def __init__(self):
        self.modelist = {
        'lr' : [{'name' : 'Logistic Regression'}, LogisticRegression()],
        'svm' : [{'name' : 'SVM'}, SVC(class_weight='balanced', probability=True)],
        'dt' : [{'name' : 'Decision Tree'}, DecisionTreeClassifier(criterion='entropy', max_depth=6, max_leaf_nodes=20, min_samples_leaf=10, min_samples_split=20)],
        'rf' : [{'name' : 'Random Forest'}, RandomForestClassifier(random_state=42, verbose=1)],
        'xgb' : [{'name' : 'Xgboost'}, XGBClassifier(objective='binary:logistic', eval_metric="logloss", nthread=-1)], 
        'knn' : [{'name' : 'KNN'}, KNeighborsClassifier()],
        'ann' : [{'name' : 'ANN'}, neural_network.MLPClassifier(activation='tanh', alpha=0.1, batch_size=5, beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08, hidden_layer_sizes=7, learning_rate='constant', learning_rate_init=0.01, max_iter=1000, momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True, power_t=0.5, random_state=None, shuffle=True, solver='lbfgs', tol=0.0001, validation_fraction=0.1, verbose=True, warm_start=False)]
        }