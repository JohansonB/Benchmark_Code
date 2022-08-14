import argparse
import time

import darts
import numpy as np
import pandas
import pandas as pd
from darts import TimeSeries
from darts.models import NBEATSModel, TCNModel, BlockRNNModel, TransformerModel, TFTModel


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--algo', type=str)
    parser.add_argument('--output_len', type=int)



    # parse the arguments
    args = parser.parse_args()

    return args

def main():
    from sktime.datasets import load_airline
    args = parse_args()
    runTime, all_horizons = dartsAllHorizon()
    windowSize, y_pred = all_horizons[0]
    #print(np.squeeze(np.array(y_pred.data_array().data)).transpose())



    with open('C:/Users/41766/IdeaProjects/BrigidJones/Scripts/Python/Output/'+args.algo+'_Out.txt', 'w') as the_file:
        the_file.write(str(runTime))
        the_file.write('\n')
        for  windowSize, y_pred in all_horizons:
            the_file.write(str(windowSize))
            the_file.write('\n')
            arr = np.squeeze(np.squeeze(np.array(y_pred.data_array().data)).transpose())
            if arr.ndim>1:
                length =arr.shape[1]
            else:
                length = len(arr)
            count = 1
            for x in np.nditer(arr):
                the_file.write(str(x))
                the_file.write(' ')
                if(count == length):
                    count = 1
                    the_file.write("\n")
                else:
                    count = count+1
            the_file.write('#')
            the_file.write('\n')


def dartsAllHorizon():
    args = parse_args()
    modelName = args.algo
    train_path = args.train_path
    test_path = args.test_path
    output_len = args.output_len

    df = TimeSeries.from_series(np.genfromtxt(train_path, delimiter=","))
    test = TimeSeries.from_series(np.genfromtxt(test_path, delimiter=","))



    data_length = int(len(df))
    test_length = int(len(test))
    print(data_length)
    print(test_length)





    input_chunk = output_len*3
    if input_chunk > int(data_length*2/3):
        input_chunk = int(data_length*2/3)
    forcaster = None
    if modelName == "NBEATS":
        forecaster = NBEATSModel(input_chunk, output_len)
    if modelName == "TCNM":
        kernel_size = 3
        if input_chunk<=kernel_size:
            kernel_size = input_chunk-1
        if output_len>input_chunk:
            input_chunk = output_len+1
        forecaster = TCNModel(input_chunk, output_len, kernel_size=kernel_size)
    if modelName == "BlockRNN":
        forecaster = BlockRNNModel(input_chunk, output_len, model='LSTM')
    if modelName == "TransformerModel":
        forecaster = TransformerModel(input_chunk, output_len)
    if modelName == "TFT":
        forecaster = TFTModel(input_chunk, output_len,add_relative_index=True)

    start = time.perf_counter()

    forecaster.fit(df)
    df_OG = df

    end = time.perf_counter()
    runTime = end - start

    all_preds = []

    for windowSize in range(1,test_length+1):
        df = df_OG
        iterer = float(test_length) / windowSize
        horizon = windowSize
        update = True

        if int(iterer) < iterer:
            iterer += 1

        iterer = int(iterer)

        y_pred = None

        for i in range(iterer):

            if (i + 1) * horizon > test_length:
                horizon = test_length - i * horizon
                update = False


            forecast = forecaster.predict(horizon, series=df)
            if y_pred is None:
                y_pred = forecast
            else:
                y_pred = y_pred.append(forecast)

            if update:
                df = df.append(test[i * windowSize:i * windowSize + windowSize])

        all_preds.append((windowSize,y_pred))

    return runTime, all_preds

if __name__ == '__main__':
    main()



