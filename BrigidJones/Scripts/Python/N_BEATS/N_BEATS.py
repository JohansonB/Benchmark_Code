import argparse
import time

import numpy as np
import pandas as pd
from NBEATS import NeuralBeats

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--dataroot', type=str)
    parser.add_argument('--testroot', type=str)
    parser.add_argument('--windowSize', type=int)

    # parse the arguments
    args = parser.parse_args()

    return args



def main():
    args = parse_args()




    data = pd.DataFrame(np.genfromtxt(args.dataroot, delimiter=","))
    y = pd.DataFrame(np.genfromtxt(args.testroot, delimiter=","))
    length, *_ = y.shape


    #train_percent = len(data)/float(len(data)+1)
    #data = data.append(pd.DataFrame(np.array([0])))



    y = pd.concat([data.iloc[-args.windowSize * 3:], y])
    y.index = list(range(0, length+args.windowSize*3))

    model = NeuralBeats(data=data.values, forecast_length=args.windowSize, train_percent=float(1))
    start = time.perf_counter()
    model.fit(verbose=False, plot=False)

    iterer = float(length) / args.windowSize

    if int(iterer) < iterer:
        iterer += 1

    iterer = int(iterer)
    y_pred = []
    for i in range(iterer):
        y_pred.extend(list(model.predict(np.array(y.iloc[i * args.windowSize:i * args.windowSize + args.windowSize * 3]))))
    y_pred = y_pred[:length]
    y_pred = [item for sublist in y_pred for item in sublist]

    end = time.perf_counter()
    runTime = end - start


    with open('Scripts/Python/Output/N_BEATS_Out.txt', 'w') as the_file:
        the_file.write(str(runTime))
        the_file.write('\n')
        for x in y_pred:
            the_file.write(str(x))
            the_file.write(' ')
        the_file.write('\n')


if __name__ == '__main__':
    main()


