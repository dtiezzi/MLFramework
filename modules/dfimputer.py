from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

class Impute:

    def __init__(self, df):
        self.df = df

    def inputate(self):
        imp = IterativeImputer(max_iter=10, verbose=0)
        imp.fit(self.df)
        imputed_df = imp.transform(self.df)
        self.df = pd.DataFrame(imputed_df, columns=self.df.columns)