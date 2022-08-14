import argparse
import time

import numpy as np
import pandas
import pandas as pd
import itertools
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.bats import BATS
from sktime.forecasting.croston import Croston
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.theta import ThetaForecaster
from sktime.forecasting.structural import UnobservedComponents
from sktime.forecasting.arima import ARIMA


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--testroot', type=str)
    parser.add_argument('--windowSize', type=int)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--d', type=int, default=0)
    parser.add_argument('--q', type=int, default=2)

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    from sktime.datasets import load_airline
    args = parse_args()
    runTime, y_pred = SKTime()
    #y_pred = list(itertools.chain(*y_pred))


    with open('Scripts/Python/Output/SKTime'+args.algo+'Out.txt', 'w') as the_file:
        the_file.write(str(runTime))
        the_file.write('\n')
        for x in y_pred:
            the_file.write(str(x))
            the_file.write(' ')
        the_file.write('\n')

def SKTime():
    args = parse_args()
    windowSize = args.windowSize
    modelName = args.algo
    dataroot = args.dataroot
    testroot = args.testroot

    x = pd.Series(np.genfromtxt(dataroot, delimiter=","))
    y = pd.Series(np.genfromtxt(testroot, delimiter=","))



    length, *_ = y.shape




    forcaster = None
    if modelName == "ExponentialSmoothing":
        forecaster = ExponentialSmoothing()
    if modelName == "AutoARIMA":
        forecaster = AutoARIMA(suppress_warnings=True)
    if modelName == "AutoETS":
        forecaster = AutoETS(auto=True)
    if modelName == "BATS":
        forecaster = BATS()
    if modelName == "Croston":
        forecaster = Croston()
    if modelName == "TBATS":
        forecaster = TBATS()
    if modelName == "ThetaForecaster":
        forecaster = ThetaForecaster()
    if modelName == "UnobservedComponents":
        forecaster = UnobservedComponents()
    if modelName == "ARIMA":
        forecaster = ARIMA(order=(args.p, args.d, args.q))

    start = time.perf_counter()



    forecaster.fit(x)



    y_pred = []

    offset = 0
    #offset = 1
    #forecaster.update(y=y.iloc[0], update_params=False)
    #y_pred.append(y.iloc[0].values[0])
    #length -= 1



    iterer = float(length) / windowSize
    horizon = windowSize
    update = True

    if int(iterer) < iterer:
        iterer += 1

    iterer = int(iterer)

    for i in range(iterer):

        if(i + 1)*horizon > length:
            horizon = length - i * horizon
            update = False

        y_pred.extend(list(forecaster.predict(fh=list(range(1, horizon + 1)))))
        if update:
            forecaster.update(y=y.iloc[i * windowSize+offset:i * windowSize + windowSize+offset], update_params=False)
    end = time.perf_counter()
    runTime = end - start
    return runTime, y_pred


if __name__ == '__main__':
    main()




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
