import pandas_ta
import pandas_ta as pta
import yfinance as yf

class StockData:
    def __init__(self, name, start_date, end_date, interval, stock, momentum):
        self.stock_data = yf.download(name, start=start_date, end=end_date, interval=interval)
        self.stock_data["RSI"] = pta.rsi(self.stock_data["Close"], length=14)
        self.stock_data["ADX"] = pta.adx(self.stock_data["High"], self.stock_data["Low"], self.stock_data["Close"], length=7)["ADX_7"]
        self.stock_data["MACD"] = pta.macd(self.stock_data["Close"], fast=4, slow=12, signal=3)["MACDs_4_12_3"]
        self.stock_data["CCI"] = pta.cci(self.stock_data["High"], self.stock_data["Low"], self.stock_data["Close"], length=14)
        self.momentum = momentum
        self.time_stamp = 1
        self.stock = stock

    def update_stock(self):
        self.time_stamp += 1
        stock_now = self.stock_data[self.time_stamp]
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
                     self.time_stamp, rsi, adx, cci, macd, self.momentum)
