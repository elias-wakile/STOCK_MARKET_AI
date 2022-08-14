import math
import numpy as np
import pandas as pd
import tensorflow as tf
import pandas_datareader as data_reader
import talib
from ta.trend import ADXIndicator
from sklearn.model_selection import train_test_split
from Stock import Stock
from StockData import StockData
import keras.backend as K
from tqdm import tqdm_notebook, tqdm
from collections import deque


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))



def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


def dataset_loader(stock_name):
    dataset = data_reader.DataReader(stock_name, data_source="yahoo")
    tmp = dataset

    dataset['CCI'] = CCI(tmp, 20)
    dataset['MACD'] = MACD(tmp)
    # start_date = str(dataset.index[0]).split()[0]
    # end_date = str(dataset.index[1]).split()[0]
    reversed_df = dataset.iloc[::-1]
    dataset["RSI"] = talib.RSI(reversed_df["Close"], 14)
    # close = dataset['Close']
    dataset["ADX"] = ADX(tmp) # todo: erase the warning
    return dataset[27:]


def ADX(dataset):
    dataset['Adj Open'] = dataset.Open * dataset['Adj Close'] / dataset['Close']
    dataset['Adj High'] = dataset.High * dataset['Adj Close'] / dataset['Close']
    dataset['Adj Low'] = dataset.Low * dataset['Adj Close'] / dataset['Close']
    dataset.dropna(inplace=True)
    adxI = ADXIndicator(dataset['Adj High'], dataset['Adj Low'], dataset['Adj Close'], 14, False)
    dataset['pos_directional_indicator'] = adxI.adx_pos()
    dataset['neg_directional_indicator'] = adxI.adx_neg()
    dataset['ADX'] = adxI.adx()
    dataset.tail()
    return dataset["ADX"]


def MACD(dataset):
    # Get the 26-day EMA of the closing price
    k = dataset['Close'].ewm(span=12, adjust=False, min_periods=12).mean()
    # Get the 12-day EMA of the closing price
    d = dataset['Close'].ewm(span=26, adjust=False, min_periods=26).mean()
    # Subtract the 26-day EMA from the 12-Day EMA to get the MACD
    macd = k - d
    return macd


def RSI(data, window=14, adjust=False):
    delta = data['Close'].diff(1).dropna()
    loss = delta.copy()
    gains = delta.copy()

    gains[gains < 0] = 0
    loss[loss > 0] = 0

    gain_ewm = gains.ewm(com=window - 1, adjust=adjust).mean()
    loss_ewm = abs(loss.ewm(com=window - 1, adjust=adjust).mean())

    RS = gain_ewm / loss_ewm
    RSI = 100 - 100 / (1 + RS)

    return RSI


def CCI(data, ndays=20):
    TP = (data['High'] + data['Low'] + data['Close']) / 3
    CCI = pd.Series(
        (TP - TP.rolling(window=ndays, center=False).mean()) / (0.015 * TP.rolling(window=ndays, center=False).std()),
        name='CCI')
    return CCI


def state_creator(dataset, timestep):
    data = dataset.iloc[timestep]
    state = [data["Adj Close"]]
    state.append(data["MACD"])
    state.append(data["RSI"])
    state.append(data["CCI"])
    state.append(data["ADX"])
    return np.array([state])


def run_trader(trader, data, batch_size):
    # trader.memory.clear()  # todo: I add this
    data_samples = data.shape[0] - 1
    state = state_creator(data, 0)
    total_profit = 0
    trader.inventory = []
    done = False

    for t in tqdm(range(data_samples)):

        action = trader.trade(state)
        next_state = state_creator(data, t + 1)
        reward = 0
        if action == 1:  # Buying
            trader.inventory.append(data['Close'][t])
            print("AI Trader bought: ", stocks_price_format(data['Close'][t]))
        elif action == 2 and len(trader.inventory) > 0:  # Selling
            buy_price = trader.inventory.pop(0)

            reward = data['Close'][t] - buy_price  # todo: may need her max with 0
            total_profit += data['Close'][t] - buy_price
            print("AI Trader sold: ", stocks_price_format(data['Close'][t]),
                  " Profit: " + stocks_price_format(data['Close'][t] - buy_price))

        if t == data_samples - 1: # todo :
            done = True

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")

        if len(trader.memory) > batch_size:  # todo: I add and t % (batch_size / 2) == 0  mey mastic
            trader.batch_train(batch_size)


if __name__ == "__main__":
    # GPUtil.showUtilization()
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    # print(tf.test.gpu_device_name())
    stock_name = "AAPL"
    stock_o = Stock(stock_name)
    stock_data = StockData(stock_name, "2020-06-02", "max", "2022-06-02", stock_o)
    a = 7
    # data = dataset_loader(stock_name)
    # window_size = 5  # 4
    # episodes = 5
    # train, test = train_test_split(data, test_size=0.5, shuffle=False)
    # budget = 1000
    # batch_size = 16 # 64
    # data_samples = train.shape[0] - 1
    # trader = AI_Trader(window_size)
    # for episode in range(1, episodes + 1):
    #     shares = 0
    #     print("Episode: {}/{}".format(episode, episodes))
    #     # state = state_creator(data, 0, window_size + 1)
    #     run_trader(trader, train, batch_size)
    #
    #     if episode % 5 == 0:
    #         trader.model.save("ai_trader_{}.h5".format(episode))
    # run_trader(trader, test, batch_size)

