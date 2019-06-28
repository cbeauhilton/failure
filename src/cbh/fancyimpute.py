from fancyimpute import SoftImpute
import pandas as pd
from cbh.exhandler import exhandler

class SoftImputeDf(SoftImpute):
    """DataFrame Wrapper around SoftImpute"""

    def __init__(self, shrinkage_value=None, convergence_threshold=0.001,
                 max_iters=100,max_rank=None,n_power_iterations=1,init_fill_method="zero",
                 min_value=None,max_value=None,normalizer=None,verbose=True):

        super(SoftImputeDf, self).__init__(shrinkage_value=shrinkage_value, 
                                           convergence_threshold=convergence_threshold,
                                           max_iters=max_iters,max_rank=max_rank,
                                           n_power_iterations=n_power_iterations,
                                           init_fill_method=init_fill_method,
                                           min_value=min_value,max_value=max_value,
                                           normalizer=normalizer,verbose=False)



    def fit_transform(self, X, y=None):

        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        cats = list(X.select_dtypes(include='category'))
        for col in X.columns:
            if X[col].isnull().sum() < 10:
                X[col].fillna(0.0, inplace=True)

        z = super(SoftImputeDf, self).fit_transform(X.values)
        df = pd.DataFrame(z, index=X.index, columns=X.columns)
        try:
            df[cats] = df[cats].astype('category')
        except:
            print("Could not reset categorical")
            pass
        # return pd.DataFrame(z, index=X.index, columns=X.columns)
        return df

    def fit_transform_cv(self, X, y=None):

        assert isinstance(X, pd.DataFrame), "Must be pandas dframe"
        cats = list(X.select_dtypes(include='category'))
        for col in X.columns:
            if X[col].isnull().sum() < 10:
                X[col].fillna(0.0, inplace=True)

        z = super(SoftImputeDf, self).fit_transform(X.values)
        df = pd.DataFrame(z, index=X.index, columns=X.columns)
        # return pd.DataFrame(z, index=X.index, columns=X.columns)
        return df