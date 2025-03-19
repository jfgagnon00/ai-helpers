import numpy as np
import pandas as pd

from sklearn.utils.validation import check_is_fitted

class LogisticRegressionBoundaryAdapter:
    """
    Adapteur permettant d'afficher les frontieres de decision 
    d'un LogisticRegression binomial avec plot_decision_boundaries_2d
    """
    def __init__(self, estimator):
        self._estimator = estimator
        try:
            check_is_fitted(estimator)
            self._is_fitted = True
        except:
            self._is_fitted = False

    def x_from(self, X, Y):
        "Adapte les variables pour call a plot_decision_boundaries_2d"
        if isinstance(X, pd.DataFrame):
            return pd.concat([X, Y], axis=1)

        return np.c_[X, Y]

    def predict(self, X):
        "Overwrite evaluation predict en '2D'"
        return self._estimator.predict( LogisticRegressionBoundaryAdapter._adapt(X) )

    def predict_proba(self, X):
        "Overwrite evaluation predict_proba en '2D'"
        return self._estimator.predict_proba( LogisticRegressionBoundaryAdapter._adapt(X) )

    def __getattr__(self, attr):
        "Redirige toutes les proprietes non 'overwritten' a l'estimateur original"
        return getattr(self._estimator, attr)
    
    def __sklearn_is_fitted__(self):
        return self._is_fitted
    
    @staticmethod
    def _adapt(X):
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, :-1]
        
        return X[:, :-1]
