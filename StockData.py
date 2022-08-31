import math
import numpy as np
import pandas
import pandas_ta as pta
import yfinance as yf


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


class StockData:
    def __init__(self, name, start_date, interval, end_date, stock, momentum=10e-1):
        """
        This function initializes the stock data object
        :param name: the stock name
        :param start_date: the start date
        :param end_date: the end date
        :param interval: the time interval
        :param stock: the stock object
        :param momentum: the momentum of the acceleration
        """
        tic = yf.Ticker(name)
        self.stock_data = tic.history(start=start_date, interval=interval, end=end_date)
        self.stock_data["DATE"] = pandas.DatetimeIndex.to_pydatetime(self.stock_data.index)
        self.row_len = self.stock_data.shape[0]
        self.stock_data["RSI"] = pta.rsi(self.stock_data["Close"], length=14)
        self.stock_data["ADX"] = \
            pta.adx(self.stock_data["High"], self.stock_data["Low"], self.stock_data["Close"],
                    length=7)["ADX_7"]
        self.stock_data["MACD"] = pta.macd(self.stock_data["Close"], fast=4, slow=12, signal=3)[
            "MACDs_4_12_3"]
        self.stock_data["CCI"] = pta.cci(self.stock_data["High"], self.stock_data["Low"],
                                         self.stock_data["Close"],
                                         length=14)
        self.momentum = momentum
        self.time_stamp = 14
        self.stock = stock
        self.max_time_stamp = self.stock_data.shape[0]
        self.new_day = True
        self.update_stock()
        self.new_day = False

    def get_state(self):
        data_current = self.stock_data.iloc[self.time_stamp]
        data_previous = self.stock_data.iloc[self.time_stamp - 1]
        state = [sigmoid(data_current["Close"] - data_previous["Close"]),
                 sigmoid(data_current["MACD"] - data_previous["MACD"]),
                 sigmoid(data_current["RSI"] - data_previous["RSI"]),
                 sigmoid(data_current["CCI"] - data_previous["CCI"]),
                 sigmoid(data_current["ADX"] - data_previous["ADX"])]
        for i in range(1, 3):
            current = self.stock_data.iloc[self.time_stamp - i]["Close"]
            previous = self.stock_data.iloc[self.time_stamp - i - 1]["Close"]
            state.append(sigmoid(current - previous))
        return np.array([state])

    def update_stock(self):
        """
        This function updates the stock this object modelizes the data
        """
        self.time_stamp += 1
        perv_date = self.stock_data.iloc[self.time_stamp - 1]["DATE"]
        if self.is_end_day():
            return False
        stock_now = self.stock_data.iloc[self.time_stamp]
        cure_date = stock_now["DATE"]
        new_day = False
        if perv_date != cure_date:
            new_day = True
        low_price = stock_now["Low"]
        high_price = stock_now["High"]
        open_price = stock_now["Open"]
        close_price = stock_now["Close"]
        volume = stock_now["Volume"]
        rsi = stock_now["RSI"]
        adx = stock_now["ADX"]
        macd = stock_now["MACD"]
        cci = stock_now["CCI"]
        self.stock.update(volume, low_price, high_price, open_price, close_price,
                          self.time_stamp, rsi, adx, cci, macd, new_day, self.momentum)
        return True

    def is_end_day(self):
        """
        check if we get over all the daily data
        :return: Ture in case we get over all the data
        """
        if self.time_stamp >= self.max_time_stamp - 1:
            return True
        return False
