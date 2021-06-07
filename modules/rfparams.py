import numpy as np

class Rfparams:

    def __init__(self):
        # Number of trees in random forest
        self.n_estimators = [int(x) for x in np.linspace(start = 30, stop = 500, num = 5)]
        # Number of features to consider at every split
        self.max_features = ['log2', 'sqrt']
        # Maximum number of levels in tree
        self.max_depth = [int(x) for x in np.linspace(5, 20, num = 5)]
        # Minimum number of samples required to split a node
        self.min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        self.min_samples_leaf = [2, 4]
        # Method of selecting samples for training each tree
        self.bootstrap = [True, False]
        