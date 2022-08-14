import argparse
import time

import numpy as np
import pandas as pd
from darts import TimeSeries

from darts.models import NBEATSModel, TCNModel, BlockRNNModel, FFT, TransformerModel, AutoARIMA, ARIMA, \
    ExponentialSmoothing, TFTModel, FourTheta, BATS, TBATS, RandomForest, LightGBMModel
from darts.utils.utils import SeasonalityMode, ModelMode, TrendMode
from pytorch_lightning.callbacks import EarlyStopping


def rescale(df):
    minimum = np.min(df)
    df = (df - minimum)
    maximum = np.max(df)
    df = df / maximum

    return minimum, maximum, df


def inverse_rescale(df, minimum, maximum):
    return df * maximum + minimum


def rescale_row_wise(df):
    if df.ndim == 1:
        return rescale(df)
    else:
        minimum = []
        maximum = []
        for i in range(len(df)):
            a, b, c = rescale(df[i])
            df[i] = c;
            minimum.append(a)
            maximum.append(b)
        return minimum, maximum, df


def inverse_rescale_row_wise(df, minimum, maximum):
    if df.ndim == 1:
        return inverse_rescale(df, minimum, maximum)
    else:
        for i in range(len(df)):
            df[i] = inverse_rescale(df[i], minimum[i], maximum[i])
        return df


def parse_args():
    """Parse arguments."""
    # Parameters settings
    parser = argparse.ArgumentParser()
    #ALL
    parser.add_argument('--normalize',type = str, default = "True")
    # Dataset setting
    parser.add_argument('--train_path', type=str)
    parser.add_argument('--output_len', type=int)
    parser.add_argument('--input_chunk', type=int, default=None)
    parser.add_argument('--test_path', type=str)
    parser.add_argument('--p', type=int, default=2)
    parser.add_argument('--d', type=int, default=0)
    parser.add_argument('--q', type=int, default=2)
    parser.add_argument('--periods', type=int, default=None)
    parser.add_argument('--algo', type=str)
    # Theta input
    parser.add_argument('--theta', type=int, default=2);
    # TCNM input
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--num_filters', type=int, default=64)
    parser.add_argument('--dilation_base', type=int, default=2)
    # FFT input
    parser.add_argument('--nb_terms', type=int, default=3)
    parser.add_argument('--trend', type=str, default=None)
    parser.add_argument('--trend_poly_degree', type=int, default = 1)
    # block RNN input
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--n_rnn_layers', type=int, default=1)
    # transformer model input
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dim_feedforward', type=int, default=512)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=64)
    # n_beats
    parser.add_argument('--num_stacks', type=int, default=30)
    parser.add_argument('--num_layers_N_BEATS', type=int, default=4)
    parser.add_argument('--layer_widths', type=int, default=256)
    # RandomForest
    parser.add_argument('--lag', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=10)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--output_chunk_length', type=int, default=1)
    #ETS
    parser.add_argument('--trend_ETS', type=str, default="NONE")
    #TBATS
    parser.add_argument('--periods_TBATS', nargs="+", type=int, default = None)
    parser.add_argument('--trend_TBATS', type=str, default = "False")
    # parse the arguments
    args = parser.parse_args()

    return args

def str_to_bool(str):
    if str == "False":
        return False
    if str == "True":
        return True
def main():
    from sktime.datasets import load_airline
    runTime, arr = darts()
    args = parse_args()

    with open('Scripts/Python/Output/' + args.algo + '_Out.txt', 'w') as the_file:

        the_file.write(str(runTime))
        the_file.write('\n')
        if arr.ndim > 1:
            length = arr.shape[1]
        else:
            length = len(arr)
        count = 1
        for x in arr.flatten():
            the_file.write(str(x))
            the_file.write(' ')
            if (count == length):
                count = 1
                the_file.write("\n")
            else:
                count = count + 1


def darts():
    my_stopper = EarlyStopping(
        monitor="train_loss",
        patience=5,
        min_delta=0.000000005,
        mode='min',
    )
    args = parse_args()



    modelName = args.algo
    train_path = args.train_path
    test_path = args.test_path
    input_chunk = args.input_chunk
    # TCNM variables
    kernel_size = args.kernel_size
    num_filters = args.num_filters
    dilation_base = args.dilation_base

    #transform ETS variable to enum type since this somehow doesnt work over command line
    args.trend_ETS = ModelMode[args.trend_ETS]
    args.normalize = str_to_bool(args.normalize)
    args.trend_TBATS = str_to_bool(args.trend_TBATS)


    #if there is a -1 in the periods list this indicates that the periods shell be set o NONE
    if -1 in args.periods_TBATS:
        args.periods_TBATS = None

    if args.normalize:
        minimum, maximum, s = rescale_row_wise(np.genfromtxt(train_path, delimiter=","))
    else:
        s = np.genfromtxt(train_path, delimiter=",")




    df = TimeSeries.from_series(s.T)
    test = TimeSeries.from_series(np.genfromtxt(test_path, delimiter=",").T)

    data_length = int(len(df))

    test_length = int(len(test))
    output_len = args.output_len

    if input_chunk is None:
        input_chunk = output_len * 3
    if input_chunk > int(data_length / 2):
        input_chunk = int(data_length / 2)
    if modelName == "NBEATS":
        forecaster = NBEATSModel(input_chunk, output_len, generic_architecture=True, num_stacks=args.num_stacks,
                                 num_layers=args.num_layers_N_BEATS, layer_widths=args.layer_widths,
                                 pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True,
                                                    "callbacks": [my_stopper]})
    if modelName == "TCNM":
        if input_chunk <= kernel_size:
            kernel_size = input_chunk - 1
        if output_len > input_chunk:
            input_chunk = output_len + 1
        forecaster = TCNModel(input_chunk, output_len, num_filters=num_filters, weight_norm=True,
                              dilation_base=dilation_base, kernel_size=kernel_size,
                              pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True,
                                                 "callbacks": [my_stopper]})
    if modelName == "BlockRNN":
        forecaster = BlockRNNModel(input_chunk, output_len, hidden_size=args.hidden_size, model='LSTM',
                                   n_rnn_layers=args.n_rnn_layers,
                                   pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True,
                                                      "callbacks": [my_stopper]})
    if modelName == "TransformerModel":
        forecaster = TransformerModel(input_chunk, output_len, d_model=args.d_model, nhead=args.nhead,
                                      num_encoder_layers=args.num_layers, num_decoder_layers=args.num_layers,
                                      dim_feedforward=args.dim_feedforward,
                                      pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True,
                                                         "callbacks": [my_stopper]})
    if modelName == "TFT":
        forecaster = TFTModel(input_chunk, output_len, hidden_size=64, add_relative_index=True,
                              pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True,
                                                 "callbacks": [my_stopper]})

    if modelName == "FFT":
        forecaster = FFT(nr_freqs_to_keep=args.nb_terms, trend_poly_degree=args.trend_poly_degree, trend=args.trend)
    if modelName == "AutoARIMA":
        forecaster = AutoARIMA(max_p=200, max_q=200)
    if modelName == "Theta":
        forecaster = FourTheta(theta=args.theta, trend_mode=TrendMode.LINEAR, season_mode=SeasonalityMode.ADDITIVE,
                               seasonality_period=args.periods)
    if modelName == "ETS":
        if args.periods is None:
            forecaster = ExponentialSmoothing(trend=args.trend_ETS, seasonal=SeasonalityMode.ADDITIVE)
        else:
            forecaster = ExponentialSmoothing(seasonal_periods=args.periods, trend=args.trend_ETS)
    if modelName == "ARIMA":
        forecaster = ARIMA(p=args.p, d=args.d, q=args.q)
    if modelName == "BATS":
        forecaster = BATS(seasonal_periods=args.periods_TBATS, use_trend=args.trend_TBATS)
    if modelName == "TBATS":
        forecaster = TBATS(seasonal_periods=args.periods_TBATS, use_trend=args.trend_TBATS)
    if modelName == "RandomForest":
        forecaster = RandomForest(lags=args.lag, max_depth=args.max_depth, n_estimators=args.n_estimators,
                                  output_chunk_length=args.output_chunk_length)
    if modelName == "GBM":
        forecaster = LightGBMModel(lags=args.lag, max_depth=args.max_depth, n_estimators=args.n_estimators,
                                   output_chunk_length=args.output_chunk_length)

    start = time.perf_counter()

    forecaster.fit(df)

    y_pred = forecaster.predict(test_length)

    end = time.perf_counter()
    runTime = end - start
    if args.normalize:
        y_pred = inverse_rescale_row_wise(np.squeeze(np.squeeze(np.array(y_pred.data_array().data)).transpose()),
                                          minimum, maximum)
    else:
        y_pred = np.squeeze(np.squeeze(np.array(y_pred.data_array().data)).transpose())
    return runTime, y_pred


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
