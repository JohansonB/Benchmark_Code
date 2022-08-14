"""
Temporal Regularized Matrix Factorization
"""

# Author: Alexander Semenov <alexander.s.semenov@yandex.ru>

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import time


class trmf:
    """Temporal Regularized Matrix Factorization.

    Parameters
    ----------
    
    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.
    
    K : int
        Length of latent embedding dimension
    
    lambda_f : float
        Regularization parameter used for matrix F.
    
    lambda_x : float
        Regularization parameter used for matrix X.
    
    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    
    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int
        Number of iterations of updating matrices F, X and W.

    F_step : float
        Step of gradient descent when updating matrix F.

    X_step : float
        Step of gradient descent when updating matrix X.

    W_step : float
        Step of gradient descent when updating matrix W.


    Attributes
    ----------

    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.

    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.

    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.
    """

    def __init__(self, lags, K, lambda_f, lambda_x, lambda_w, alpha, eta, max_iter=1000, 
                 F_step=0.0001, X_step=0.0001, W_step=0.0001, windowSize = 1, data = None):
        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        self.max_iter = max_iter
        self.F_step = F_step
        self.X_step = X_step
        self.W_step = W_step
        self.windowSize = windowSize
        self.data = data
        self.dataProjection = None
        
        self.W = None
        self.F = None
        self.X = None


    def fit(self, train, resume=False):
        """Fit the TRMF model according to the given training data.

        Model fits through sequential updating three matrices:
            -   matrix self.F;
            -   matrix self.X;
            -   matrix self.W.
            
        Each matrix updated with gradient descent.

        Parameters
        ----------
        train : ndarray, shape (n_timeseries, n_timepoints)
            Training data.

        resume : bool
            Used to continue fitting.

        Returns
        -------
        self : object
            Returns self.
        """

        if not resume:
            self.Y = train
            mask = np.array((~np.isnan(self.Y)).astype(int))
            self.mask = mask
            self.Y[self.mask == 0] = 0.
            self.N, self.T = self.Y.shape
            ''' #Correct init
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)
            '''

                #test_init
            self.W = np.full((self.K, self.L),0.5) / self.L
            self.F = np.full((self.N, self.K),0.5)
            self.X = np.full((self.K, self.T),0.5)


        for _ in range(self.max_iter):
            self._update_F(step=self.F_step)
            self._update_X(step=self.X_step)
            self._update_W(step=self.W_step)


    def predict(self, h):
        """Predict each of timeseries h timepoints ahead.

        Model evaluates matrix X with the help of matrix W,
        then it evaluates prediction by multiplying it by F.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """

        X_preds = self._predict_X(h)
        return np.dot(self.F, X_preds)


    def _predict_X(self, h):
        """Predict X h timepoints ahead.

        Evaluates matrix X with the help of matrix W.

        Parameters
        ----------
        h : int
            Number of timepoints to forecast.

        Returns
        -------
        X_preds : ndarray, shape (self.K, h)
            Predictions of timepoints latent embeddings.
        """
        print(self.F)
        self.dataProjection = np.dot(np.linalg.pinv(self.F),self.data)
        X_preds = np.zeros((self.K, h))
        X_adjusted = np.hstack([self.X, X_preds])
        X_temp = np.copy(X_adjusted)
        for t in range(self.T, self.T + h):
            if((t-self.T)%self.windowSize==0):
                for i in range(1,self.windowSize+1):
                    X_temp[:,(t-i)]  = self.dataProjection[:,(t-i)]


            for l in range(self.L):
                lag = self.lags[l]
                #if t-lag >= 0:
                X_temp[:, t] += X_temp[:, t - lag] * self.W[:, l]
                X_adjusted[:, t] += X_temp[:, t - lag] * self.W[:, l]
        return X_adjusted[:, self.T:]

    def impute_missings(self):
        """Impute each missing element in timeseries.

        Model uses matrix X and F to get all missing elements.

        Parameters
        ----------

        Returns
        -------
        data : ndarray, shape (n_timeseries, T)
            Predictions.
        """
        data = self.Y
        data[self.mask == 0] = np.dot(self.F, self.X)[self.mask == 0]
        return data


    def _update_F(self, step, n_iter=1):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.F -= step * self._grad_F()


    def _update_X(self, step, n_iter=1):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.X -= step * self._grad_X()


    def _update_W(self, step, n_iter=1):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()


    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        return - 2 * np.dot((self.Y - np.dot(self.F, self.X)) * self.mask, self.X.T) + 2 * self.lambda_f * self.F


    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.

        grad_T_x = z_1 + z_2
        temp = - 2 * np.dot(self.F.T, self.mask * (self.Y - np.dot(self.F, self.X))) + self.lambda_x * grad_T_x + self.eta * self.X
        return - 2 * np.dot(self.F.T, self.mask * (self.Y - np.dot(self.F, self.X))) + self.lambda_x * grad_T_x + self.eta * self.X


    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x - self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)




'''lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.

    K : int
        Length of latent embedding dimension

    lambda_f : float
        Regularization parameter used for matrix F.

    lambda_x : float
        Regularization parameter used for matrix X.

    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.

    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    max_iter : int
        Number of iterations of updating matrices F, X and W.

    F_step : float
        Step of gradient descent when updating matrix F.

    X_step : float
        Step of gradient descent when updating matrix X.

    W_step : float
        Step of gradient descent when updating matrix W.'''



def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="thesis.Models.trmf implementation")


    parser.add_argument('--dataroot', type=str, help='path to dataset')
    parser.add_argument("--lags", nargs="+",type = int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--l_f', type=float)
    parser.add_argument('--l_x', type=float)
    parser.add_argument('--l_w', type=float)
    parser.add_argument('--eta', type=float)
    parser.add_argument('--alpha', type=float)
    parser.add_argument('--max_iter', type=int)
    parser.add_argument('--F_step', type=float)
    parser.add_argument('--X_step', type=float)
    parser.add_argument('--W_step', type=float)
    parser.add_argument('--len', type=int)
    parser.add_argument('--split', type=float, default=0.7)
    parser.add_argument('--windowSize', type=int, default=1)

    # parse the arguments
    args = parser.parse_args()

    return args



def main():

    args = parse_args()

    df = pd.DataFrame(np.genfromtxt(args.dataroot, delimiter=","))


    size = args.len
    data = df.to_numpy()
    X = df.iloc[:,:size+1].to_numpy()
    Y = df.iloc[:,size+1:]
    model = trmf(
        lags = args.lags,
        K = args.k,
        lambda_f = args.l_f,
        lambda_x = args.l_x,
        lambda_w = args.l_w,
        eta = args.eta,
        F_step = args.F_step,
        X_step = args.X_step,
        W_step = args.W_step,
        alpha = args.alpha,
        windowSize = args.windowSize,
        data = data
        )

    start = time.perf_counter()

    model.fit(train = X)
    pred = model.predict(Y.shape[1])

    end = time.perf_counter()

    runTime = end - start

    with open('Scripts/Python/Output/trmf_Output.txt', 'w') as the_file:
        the_file.write(str(runTime))
        the_file.write('\n')
        for p in pred:
            for d in p:
                the_file.write(str(d))
                the_file.write(' ')
            the_file.write("\n")








if __name__ == "__main__":
    main()
