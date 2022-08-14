import argparse
import math
import time

import numpy
import numpy as np
import pandas
import pandas as pd
import sklearn
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression


def relative(S):
    return S / sum(S)


def solveQuadratic(a, b, c):
    # calculating discriminant using formula
    dis = b * b - 4 * a * c
    if(dis< 0):
        raise Exception("no real roots")
    sqrt_val = math.sqrt(dis)


    r1 = (-b + sqrt_val) / (2 * a)
    r2 = (-b - sqrt_val) / (2 * a)

    if r1>= 0:
        return r1
    if r2>=0 :
        return r2
    else:
        raise Exception("no positive root")


def project(in_vec, basis):
    _, m = basis.shape
    projection = np.zeros(len(in_vec))
    for i in range(m):
        projection = projection + in_vec.dot(basis[:,i])*basis[:,i]

    return pd.Series(projection)


class SVDME:
    def __init__(self, train_root, test_root, compression_ratio=0.8, MN_ratio=0.5, n = 0):
        start = time.perf_counter()
        self.compression_ratio = compression_ratio
        self.MN_ratio = MN_ratio

        #impute missing values with intermidiate value
        self.X = pd.Series(np.genfromtxt(train_root, delimiter=","))
        self.Y = pd.Series(np.genfromtxt(test_root, delimiter=","))
        self.prediction_length = len(self.Y)

        maximum = np.nanmax(self.X)

        minimum = np.nanmin(self.X)
        diff = 0.5 * (minimum + maximum)
        self.X = self.X.fillna(value=diff)

        self.train_length = len(self.X)



        if n == 0:
            self.n = int(solveQuadratic(1/MN_ratio, 1, -self.train_length))
            self.m = int(self.n / MN_ratio)
        else:
            self.n = n
            self.m = int((self.train_length - self.n)/self.n)

        self.forecast = pandas.Series(dtype='float64')
        page_matrices = self._init_page_matrices()
        iterations = min(self.n, self.prediction_length)
        basis = np.empty(iterations,dtype = numpy.ndarray)
        for i in range(len(page_matrices)):
            page_matrices[i], basis[i] = self._denoise(page_matrices[i])

        models = self._get_models(page_matrices)
        end = time.perf_counter()
        self.runTime = end - start

        self.preds = []

        for window_size in range(1,self.prediction_length+1):

            # initialize input to linear regression
            in_vec = self.X[-(self.n - 1):]
            # in_vec = pd.Series(page_matrices[0][:-1,-1])

            self.forecast = np.empty(self.prediction_length, dtype=float)

            for i in range(self.prediction_length):
                prediction = models[i%len(models)].predict(np.array(in_vec).reshape(1, -1))

                in_vec = in_vec.append(pandas.Series(prediction))
                #in_vec = project(in_vec.to_numpy(), basis[i % len(models)])

                #if window size is reached update the estimated X values with the real once

                self.forecast[i] = in_vec.iloc[-1]
                in_vec = in_vec[1:]
                if (i+1) % window_size == 0 and i+1 <= self.prediction_length:
                    stepper = min(window_size,len(in_vec))
                    in_vec.iloc[-stepper:] = self.Y[i+1-stepper:i+1]

            self.preds.append((window_size,self.forecast))


    def _init_page_matrices(self):
        matrixDimension = self.n * self.m
        iterations = min(self.n,self.prediction_length)
        page_matrices = np.empty(iterations,dtype = numpy.ndarray)
        for i in range(iterations):
            offset = self.train_length-matrixDimension-iterations+i

            page_matrices[iterations-i-1] = self._init_page_matrix(offset)
        return page_matrices

    def _init_page_matrix(self, offset):
        page_matrix = np.zeros((self.n, self.m))

        for M in range(self.m):
            for N in range(self.n):
                page_matrix[N][M] = self.X[M*self.n+N+offset]
        return page_matrix






    def _denoise(self,page_matrix):
        U, S, VT = np.linalg.svd(page_matrix)
        i = 1
        rS = relative(S)

        while sum(rS[:i]) <= self.compression_ratio and i<=len(rS):
            i = i + 1
        i = i - 1
        if i == 0:
            i = 1
        for j in range(len(S)):
            if j>=i:
                S[j] = 0

        smat = np.zeros((len(U),len(VT)))
        smat[:len(S),:len(S)] = np.diag(S)
        return np.dot(np.dot(U,smat),VT), U[:,:i]

    def _learn_model(self,page_matrix):
        model = LinearRegression()
        x = np.transpose(page_matrix[:self.n-1,:self.m])
        y = page_matrix[self.n-1,:self.m]
        model.fit(x, y)
        return model

    def _get_models(self, page_matrices):
        models = np.empty(len(page_matrices),dtype=sklearn.linear_model._base.LinearRegression)
        for i in range(len(page_matrices)):
            models[i] = self._learn_model(page_matrices[i])
        return models



    def plot(self,page_matrix):
        x = np.arange(0,self.n*self.m)
        y = np.zeros(self.n*self.m)
        for j in range(self.m):
            for i in range(self.n):
                y[j*self.n+i] = page_matrix[i,j]

        plt.title("Line graph")
        plt.xlabel("X axis")
        plt.ylabel("Y axis")
        plt.plot(x, y, color="red")
        plt.show()


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser(description="thesis.Models.trmf implementation")

    parser.add_argument('--train_root', type=str, help='path to dataset')
    parser.add_argument('--test_root', type=str, help='path to dataset')
    parser.add_argument('--n', type=int,default=0)
    parser.add_argument('--compression_ratio', type=float)
    parser.add_argument('--MN_ratio', type=float)


    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    args = parse_args()


    mod = SVDME(args.train_root, args.test_root, args.compression_ratio, args.MN_ratio, args.n)



    with open('Scripts\Python\ME\out.txt', 'w') as the_file:
        the_file.write(str(mod.runTime))
        the_file.write('\n')
        for tuple in mod.preds:
            windowSize, y_pred = tuple
            the_file.write(str(windowSize))
            the_file.write('\n')

            for x in y_pred:
                the_file.write(str(x))
                the_file.write(' ')
            the_file.write('#')
            the_file.write('\n')


if __name__ == "__main__":
    main()

















