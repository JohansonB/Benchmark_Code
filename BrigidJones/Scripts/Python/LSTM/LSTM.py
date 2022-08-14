import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.python import tf2
from scipy.ndimage.filters import gaussian_filter
import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



'''input:
    --dataroot:  String: filepath to csv with Timeseries stored
    --AR: Integer: AR-degree
    --split: Double: train-testSplit factor
    --lt: Boolean: "T" , "F" enable logtransformation
    --dif: Boolean: "T" , "F" enable diffrencing
    --nmbnodes: Integer: number of Nodes in Network
    --epochs: Integer: epochs
    --verbose: Integer: verbose
    '''


def create_dataset(data_series, look_back, split_frac):

    # scaling values between 0 and 1
    dates = data_series.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_series.values.reshape(-1, 1))
    data_series = pd.Series(scaled_data[:, 0])

    # creating targets and features by shifting values by 'i' number of time periods
    df = pd.DataFrame()
    for i in range(look_back + 1):
        label = ''.join(['t-', str(i)])
        df[label] = data_series.shift(i)

    df = df.dropna()

    # splitting data into train and test sets
    size = split_frac
    print(size)
    train = df[:size]
    test = df[size:]

    # creating target and features for training set
    X_train = train.iloc[:, 1:].values

    y_train = train.iloc[:, 0].values
    train_dates = train.index

    # creating target and features for test set
    X_test = test.iloc[:, 1:].values
    y_test = test.iloc[:, 0].values
    test_dates = test.index

    # reshaping data into 3 dimensions for modeling with the thesis.Models.LSTM neural net
    X_train = np.reshape(X_train, (X_train.shape[0], 1, look_back))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, look_back))

    return X_train, y_train, X_test, y_test, train_dates, test_dates, scaler


def inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler):
    # inverse 0 to 1 scaling
    train_predict = pd.Series(scaler.inverse_transform(train_predict.reshape(-1, 1))[:, 0], index=train_dates)
    y_train = pd.Series(scaler.inverse_transform(y_train.reshape(-1, 1))[:, 0], index=train_dates)

    test_predict = pd.Series(scaler.inverse_transform(test_predict.reshape(-1, 1))[:, 0], index=test_dates)
    y_test = pd.Series(scaler.inverse_transform(y_test.reshape(-1, 1))[:, 0], index=test_dates)


    return train_predict, y_train, test_predict, y_test


def lstm_model(data_series, look_back, split, lstm_params):
    np.random.seed(1)

    # creating the training and testing datasets
    X_train, y_train, X_test, y_test, train_dates, test_dates, scaler = create_dataset(data_series, look_back, split)

    # training the model
    model = Sequential()
    model.add(LSTM(lstm_params[0], input_shape=(1, look_back)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    #for i in range(lstm_params[1]):
    #    model.fit(X_train, y_train, epochs=1, batch_size=1, verbose=lstm_params[2])
    #    model.reset_states()
    model.fit(X_train, y_train, epochs=lstm_params[1], batch_size=1, verbose=lstm_params[2])
    # making predictions
    train_predict = model.predict(X_train)

    test_predict = model.predict(X_test)

    # inverse transforming results
    train_predict, y_train, test_predict, y_test = \
        inverse_transforms(train_predict, y_train, test_predict, y_test, data_series, train_dates, test_dates, scaler)

    '''# plot of predictions and actual values
    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.plot(y_test)
    plt.plot(test_predict, color='red')
    plt.show()

    # calculating RMSE metrics
    out = ""
    out += str(np.sqrt(mean_squared_error(train_predict, y_train)))
    out += " "
    out += str(np.sqrt(mean_squared_error(test_predict, y_test)))
    print(out)'''

    return train_predict, y_train, test_predict, y_test

def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()

    # Dataset setting
    parser.add_argument('--dataroot', type=str, default="C:/Users/41766/Desktop/Baechlor_Sourceroni_code/GeneratedTestData/sphist.csv")
    parser.add_argument('--AR', type=int, default=1)

    # Encoder / Decoder parameters setting
    parser.add_argument('--split', type=float, default=0.7)
    parser.add_argument('--nmbNodes', type=int, default=4)

    # Training parameters setting
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train [10, 200, 500]')
    parser.add_argument('--verbose', type=float, default=0)
    parser.add_argument('--len', type=int)

    # parse the arguments
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    #path: 'C:/Users/41766/Desktop/Baechlor_Sourceroni_code/GeneratedTestData/sphist.csv'
    sp500_ts = pd.DataFrame(np.genfromtxt(args.dataroot, delimiter=",")).transpose()
    look_back = int(args.AR)+1
    split = args.split


    nodes = args.nmbNodes
    epochs = args.epochs
    verbose = args.verbose  # 0=print no output, 1=most, 2=less, 3=least
    lstm_params = [nodes, epochs, verbose]

    start = time.perf_counter()

    train_predict, y_train, test_predict, y_test = lstm_model(sp500_ts, look_back, args.len, lstm_params)

    end = time.perf_counter()
    runTime = end-start

    print(test_predict.shape)


    with open('Scripts/Python/Output/LSTM_Output.txt', 'w') as the_file:
        the_file.write(str(runTime))
        the_file.write('\n')
        for x in test_predict:
            the_file.write(str(x))
            the_file.write(' ')
        the_file.write('\n')




if __name__ == "__main__":
 main()