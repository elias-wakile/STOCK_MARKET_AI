import pandas_ta
import pandas_ta as pta
import yfinance as yf

class StockData:
    def __init__(self, name, start_date, period, end_date, stock, momentum):
        """
        This function initializes the stock data object
        :param name: the stock name
        :param start_date: the start date
        :param end_date: the end date
        :param interval: the time interval
        :param stock: the stock object
        :param momentum: the momentum of the acceleration
        """
        self.stock_data = yf.download(name, start=start_date, period=period, end=end_date)[29:329]
        self.stock_data["RSI"] = pta.rsi(self.stock_data["Close"], length=14)
        self.stock_data["ADX"] = pta.adx(self.stock_data["High"], self.stock_data["Low"], self.stock_data["Close"], length=7)["ADX_7"]
        self.stock_data["MACD"] = pta.macd(self.stock_data["Close"], fast=4, slow=12, signal=3)["MACDs_4_12_3"]
        self.stock_data["CCI"] = pta.cci(self.stock_data["High"], self.stock_data["Low"], self.stock_data["Close"], length=14)
        self.momentum = momentum
        self.time_stamp = 0
        self.stock = stock
        self.max_time_stamp = self.stock_data.size
        self.update_new_day()

    def update_stock(self):
        """
        This function updates the stock this object modelizes the data
        """
        self.time_stamp += 1
        if self.time_stamp == self.stock_data.size:
            return 0
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
        return 1

    def update_new_day(self):
        """
        This function updates a stock to the new day
        """
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
        self.stock.update(volume, low_price, high_price, open_price,
                          close_price, self.time_stamp, rsi, adx, cci, macd,
                          self.momentum, new_day=True, momentum=10e-1)

if __name__ == '__main__':
    stock_data = yf.download("AAPL", start="2022-01-01", end="2022-01-02",
                                  interval="1m")[29:329]
    print(stock_data.size)