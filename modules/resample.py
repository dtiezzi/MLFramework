from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

class Resampling:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def smote(self, test_size=0.4):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, stratify=self.y)
        sm = SMOTE(random_state=47)
        self.X_train, self.y_train = sm.fit_resample(self.X_train, self.y_train)
        