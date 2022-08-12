from Stock import Stock
from StockMarket import StockData
import pandas_datareader.data as pdr
import numpy as np

class PortFolio:
    def __init__(self, initial_investment, stock_list, period, date_list, stock_indices):
        """
        This function is a portfolio constructor
        :param initial_investment: The initial investment in the portfolio
        :param stock_list: the list of stocks we wish to invest in
        :param period: the period of time we wish to refresh our data
        :param date_list: the list of days we want our model to work on
        :param stock_indices: A dictionary having as keys the stock names and
                              as values their indices in the state matrix
                              and prediction vector
        """
        self.stock_name_list = stock_list
        self.stocks = dict()
        self.balance = initial_investment
        self.period = period
        for stock_name in stock_list:
             self.stocks[stock_name] = Stock(stock_name)
        self.date_list = date_list
        self.stock_indices = stock_indices
        self.features_num = 18


    def next_day(self):
        """
        This function moves our portfolio to the next day
        Not needed for the Network
        """
        if len(self.date_list) >= 2:
            curr_date = self.date_list[0]
            next_date = self.date_list[1]
            self.date_list = self.date_list[1:]
            self.stock_market = {stock_name : StockData(stock_name, curr_date,
                                                        self.period, next_date,
                                                        self.stocks[stock_name], 10e-1)
                                 for stock_name in self.stock_name_list}

    def update_portfolio(self):
        """
        This function updates the portfolio for the current date
        """
        for stock_name in self.stock_name_list:
            if self.stock_market[stock_name].update_stock() == 1:
                self.next_day()

    def getBalance(self):
        """
        Gets the balance of the portfolio
        :return: The sum of balance that we have in cash and the money we could
        have if we immediately sold our stocks.
        """
        stock_balance = 0
        for stock in self.stock_name_list:
            stock_balance += self.stocks[stock].money_in_stock
        print(f"Your Portfolio currently has a balance of {self.balance}, owns"
              f"a stock of total value {stock_balance} and is of total value of"
              f"{stock_balance + self.balance}.")
        return stock_balance + self.balance

    def getState(self):
        state = np.array([len(self.stock_name_list), self.features_num])
        for stock_name in self.stock_indices.keys():
            line_index = self.stock_indices[stock_name]
            state[line_index, 0] = self.stocks[stock_name].self.daily_highest
            state[line_index, 1] = self.stocks[stock_name].self.daily_lowest
            state[line_index, 2] = self.stocks[stock_name].self.daily_precentile_acceleration
            state[line_index, 3] = self.stocks[stock_name].self.market_volume_acceleration
            state[line_index, 4] = self.stocks[stock_name].self.stock_price_acceleration
            state[line_index, 5] = self.stocks[stock_name].self.per_var
            state[line_index, 6] = self.stocks[stock_name].self.volume_var
            state[line_index, 7] = self.stocks[stock_name].self.price_var
            state[line_index, 8] = self.stocks[stock_name].self.current_price_daily_percentile
            state[line_index, 9] = self.stocks[stock_name].self.last_market_volume
            state[line_index, 10] = self.stocks[stock_name].self.last_high_price
            state[line_index, 11] = self.stocks[stock_name].self.last_low_price
            state[line_index, 12] = self.stocks[stock_name].self.last_open_price
            state[line_index, 13] = self.stocks[stock_name].self.last_close_price
            state[line_index, 14] = self.stocks[stock_name].self.ADX
            state[line_index, 15] = self.stocks[stock_name].self.MACD
            state[line_index, 16] = self.stocks[stock_name].self.CCI
            state[line_index, 17] = self.stocks[stock_name].self.RSI
        return state

    def act(self, stock_predictions):
        """
        This function executes the predictions of the Model
        :param stock_predictions: the number of shares we wanna buy of each stock
                                  the number of shares we wanna buy for AAPL
                                  will be at self.stock_indices["AAPL"]
        :return: A reward vector of the same format
        """
        results = [0] * len(self.stock_name_list)
        for stock_name in self.stock_indices.keys():
            index = self.stock_indices[stock_name]
            num_of_stocks = stock_predictions[index]
            if num_of_stocks < -self.stocks[stock_name].num_of_stocks:
                num_of_stocks = self.stocks[stock_name].num_of_stocks
            elif num_of_stocks * self.stocks[stock_name].last_low_price >= self.balance:
                num_of_stocks = int(self.balance / (self.stocks[stock_name].last_low_price))
            curr_loss = stock.trade()
            if num_of_stocks < 0:
                self.balance += num_of_stocks * self.stocks[stock_name].last_high_price
            elif num_of_stocks > 0:
                self.balance -= num_of_stocks * self.stocks[stock_name].last_low_price
            results[index] = curr_loss
        return np.array(results)