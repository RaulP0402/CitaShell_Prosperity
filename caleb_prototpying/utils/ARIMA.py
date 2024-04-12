from typing import List, Any, Dict, Tuple
import json, collections
import numpy as np
from dataclasses import dataclass
from collections import OrderedDict
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState, UserId

def least_squares(x, y):
    """
    Compute the coefficients of a linear regression model using the least squares method.

    Parameters:
        x (array-like): The independent variable(s) of the linear regression model.
        y (array-like): The dependent variable of the linear regression model.

    Returns:
        array-like: The coefficients of the linear regression model.

    Notes:
        - If the determinant of x.T @ x is not equal to 0, the function computes the coefficients using the inverse of x.T @ x.
        - If the determinant of x.T @ x is equal to 0, the function computes the coefficients using the pseudo inverse of x.T @ x.

    """
    if np.linalg.det(x.T @ x) != 0:     # if determinant is not 0
        return np.linalg.inv((x.T @ x)) @ (x.T @ y)     #computes coefficients of LR model
    return np.linalg.pinv((x.T @ x)) @ (x.T @ y) # matrix not invertible uses pusedo inverse to give coefficients for LR model


"""
Moving Average 
"""
n = 500
eps = np.random.normal(size=n) #generating random distribution
def lag_view(x, order):
    y = x.copy()  
 
    x = np.array([y[-(i + order):][:order] for i in range(y.shape[0])])

    x = np.stack(x)[::-1][order - 1: -1]
    y = y[order:] 

    return x, y


"""
Differencing 
"""

#integrated
def difference(x, d=1):
    if d == 0:
        return x
    else:
        # calculates different between ith and ith+1
        x = np.r_[x[0], np.diff(x)]
        return difference(x, d - 1)

#integrated
def undo_difference(x, d=1):
    """
    Undo Difference

    Reverses the differencing operation performed on a time series.

    Parameters:
        x (array-like): The differenced time series.
        d (int, optional): The number of times the time series was differenced. Default is 1.

    Returns:
        array-like: The original time series.

    Notes:
        The undo_difference function reverses the differencing operation performed by the difference function.
        It calculates the cumulative sum of the differenced time series.
        If d is 1, the cumulative sum is returned.
        If d is greater than 1, the function recursively applies the cumulative sum d times.

    Examples:
        >>> x = [1, 2, 3, 4]
        >>> undo_difference(x)
        [1, 3, 6, 10]
        >>> undo_difference(x, d=2)
        [1, 4, 10, 20]
    """
    if d == 1:
        return np.cumsum(x)
    else:
        x = np.cumsum(x)
        return undo_difference(x, d - 1)

"""
Linear Regression
"""
class LinearModel:
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.beta = None # stores coefficients
        self.intercept_ = None # stores intercepts
        self.coef_ = None # stores coeffecients of independent vars

    def _prepare_features(self, x):
        #takes independent variables x
        # adds column of ones to each feature
        if self.fit_intercept:
            x = np.hstack((np.ones((x.shape[0], 1)), x))
        return x

    def fit(self, x, y):
        x = self._prepare_features(x)
        self.beta = least_squares(x, y)
        if self.fit_intercept:
            self.intercept_ = self.beta[0]
            self.coef_ = self.beta[1:]
        else:
            self.coef_ = self.beta

    def predict(self, x):
        x = self._prepare_features(x)
        return x @ self.beta

    def fit_predict(self, x, y):
        self.fit(x, y)
        return self.predict(x)
    
class ARIMA(LinearModel):
    def __init__(self, q, d, p):
        """
        An ARIMA model.
        :param q: (int) Order of the Autoregressor part
        :param d: (int) Number of times the data needs to be differenced to be stationary
        :param p: (int) Order of the moving average part
        """
        super().__init__(True) # calls constructor of linearmodel
        self.q = q
        self.d = d
        self.p = p
        self.ar = None
        self.resid = None # will hold residual errors

    def prepare_features(self, x):
        if self.d > 0: # determines if differencing is needed, if so makes data stationary
            x = difference(x, self.d)

        ar_features = None
        ma_features = None

        if self.q > 0: # order of MA 
            if self.ar is None:
                self.ar = ARIMA(0, 0, self.p) # initialize ARIMA & only focus on autoregressie to fit model to data
                self.ar.fit_predict(x)
            eps = self.ar.resid # storing residuals froms fitted ar model 
            eps[0] = 0 

            # prepare moving avg features by creating lagged versions of residuals
            ma_features, _ = lag_view(np.r_[np.zeros(self.q), eps], self.q)

        # Determine the features for the AR process
        if self.p > 0:
            # prepend with zeros as there are no X_t-k in the first X_t
            ar_features = lag_view(np.r_[np.zeros(self.p), x], self.p)[0]
        
        #combining and truncating ma and ar features
        if ar_features is not None and ma_features is not None:
            n = min(len(ar_features), len(ma_features))
            ar_features = ar_features[:n]
            ma_features = ma_features[:n]
            features = np.hstack((ar_features, ma_features))
        elif ma_features is not None:
            n = len(ma_features)
            features = ma_features[:n]
        else:
            n = len(ar_features)
            features = ar_features[:n]

        return features, x[:n] #returns prepared features & x truncated to match len(features)

    def fit(self, x):
        features, x = self.prepare_features(x)
        super().fit(features, x) # calls fit from LR class to fit using prepared features
        return features

    def fit_predict(self, x):
        features = self.fit(x)
        return self.predict(x, prepared=(features))

    def predict(self, x, **kwargs):
        features = kwargs.get('prepared', None)
        if features is None:
            features, x = self.prepare_features(x)

        y = super().predict(features)
        self.resid = x - y # calc dif between actual and predcited

        return self.return_output(y)

    def return_output(self, x):
        if self.d > 0: # differenced > 0
            x = undo_difference(x, self.d) # undo difference b/c d > 0 
        return x

    def forecast(self, x, n):
        features, x = self.prepare_features(x)
        y = super().predict(features)

        # appends n zeros to end of y predictions
        # essentially making space for future predictions
        y = np.r_[y, np.zeros(n)]
        
        for i in range(n):
            feat = np.r_[y[-(self.p + n) + i: -n + i], np.zeros(self.q)]
            y[x.shape[0] + i] = super().predict(feat[None, :])
        return self.return_output(y)


