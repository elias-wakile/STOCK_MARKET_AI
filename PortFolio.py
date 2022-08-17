from Stock import Stock
from StockData import StockData
import pandas_datareader.data as pdr
from datetime import datetime
import numpy as np


class PortFolio:
    def __init__(self, initial_investment, stock_list, interval, date_list, stock_indices):
        """
        This function is a portfolio constructor
        :param initial_investment: The initial investment in the portfolio
        :param stock_list: the list of stocks we wish to invest in
        :param interval: the interval of time we wish to refresh our data
        :param date_list: the list of days we want our model to work on
        :param stock_indices: A dictionary having as keys the stock names and
                              as values their indices in the state matrix
                              and prediction vector
        """
        self.stock_name_list = stock_list
        self.stocks = dict()
        self.balance = initial_investment
        self.interval = interval
        for stock_name in stock_list:
            self.stocks[stock_name] = Stock(stock_name)
        self.date_list = date_list
        self.stock_indices = stock_indices
        self.features_num = 20  # todo: was 18 but in getState there is 20 features
        self.stock_market = {}
        self.next_day()

    def next_day(self):
        """
        This function moves our portfolio to the next day
        Not needed for the Network
        """
        if len(self.date_list) >= 2:
            curr_date = self.date_list[0]
            next_date = self.date_list[1]
            self.date_list = self.date_list[1:]
            self.stock_market = {stock_name: StockData(stock_name, curr_date,
                                                       self.interval, next_date,
                                                       self.stocks[stock_name], 10e-1)
                                 for stock_name in self.stock_name_list} # todo: what this need to be ?

    def update_portfolio(self):
        """
        This function updates the portfolio for the current date
        """
        for stock_name in self.stock_name_list:
            if not self.stock_market[stock_name].update_stock():
                self.next_day()
                break
                # self.stock_market[stock_name].update_stock()

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
              f"a stock of total value {stock_balance} and is of total value of "
              f"{stock_balance + self.balance}.") # todo: there is no update of the stock of total value
        return stock_balance + self.balance

    def getState(self):
        """
        This function gets the current state of the Portfolio
        :return: The state of the Portfolio, each line represents a stock and
        the columns represent the different values of the parameters
        """
        state = np.zeros((len(self.stock_name_list), self.features_num))
        for line_index in self.stock_indices.keys():
            stock_name = self.stock_indices[line_index]
            state[line_index, 0] = self.stocks[stock_name].daily_highest
            state[line_index, 1] = self.stocks[stock_name].daily_lowest
            state[line_index, 2] = self.stocks[stock_name].daily_precentile_acceleration
            state[line_index, 3] = self.stocks[stock_name].market_volume_acceleration
            state[line_index, 4] = self.stocks[stock_name].stock_price_acceleration
            state[line_index, 5] = self.stocks[stock_name].per_var
            state[line_index, 6] = self.stocks[stock_name].volume_var
            state[line_index, 7] = self.stocks[stock_name].price_var
            state[line_index, 8] = self.stocks[stock_name].current_price_daily_percentile
            state[line_index, 9] = self.stocks[stock_name].last_market_volume
            state[line_index, 10] = self.stocks[stock_name].last_high_price
            state[line_index, 11] = self.stocks[stock_name].last_low_price
            state[line_index, 12] = self.stocks[stock_name].last_open_price
            state[line_index, 13] = self.stocks[stock_name].last_close_price
            state[line_index, 14] = self.stocks[stock_name].ADX
            state[line_index, 15] = self.stocks[stock_name].MACD
            state[line_index, 16] = self.stocks[stock_name].CCI
            state[line_index, 17] = self.stocks[stock_name].RSI
            state[line_index, 18] = self.stocks[stock_name].num_of_stocks_owned
            state[line_index, 19] = self.stocks[stock_name].price_per_stock
        return state

    def end_data(self):
        """
        This function check if there is more data to check
        :return: True if we look over all the data else False
        """
        if len(self.date_list) == 1:
            for stock_name in self.stock_name_list:
                if not self.stock_market[stock_name].is_end_day():
                    return False
            return True
        return False

    def act(self, stock_predictions):
        """
        This function executes the predictions of the Model
        :param stock_predictions: the number of shares we wanna buy of each stock
                                  the number of shares we wanna buy for AAPL
                                  will be at self.stock_indices["AAPL"]
        :return: A reward vector of the same format
        """
        results = [0] * len(self.stock_name_list)
        real_act = [0] * len(self.stock_name_list)
        for index in self.stock_indices.keys():
            stock_name = self.stock_indices[index]
            num_of_stocks = stock_predictions[stock_name]
            if num_of_stocks < -self.stocks[stock_name].num_of_stocks_owned:
                num_of_stocks = -self.stocks[stock_name].num_of_stocks_owned
            elif num_of_stocks * self.stocks[stock_name].last_low_price >= self.balance:
                num_of_stocks = int(self.balance / self.stocks[stock_name].last_low_price)
            curr_loss = self.stocks[stock_name].trade(num_of_stocks)
            if num_of_stocks < 0:
                self.balance += -num_of_stocks * self.stocks[stock_name].last_high_price # I think this is the sell so
                # I cheng the num_of_stocks to be -num_of_stocks
            elif num_of_stocks > 0:
                self.balance -= num_of_stocks * self.stocks[stock_name].last_low_price
            results[index] = curr_loss
            real_act[index] = num_of_stocks
            if stock_predictions[stock_name] != num_of_stocks:
                print("want to do: " + str(stock_predictions[stock_name]) + " but do " + str(num_of_stocks))
        # return np.array(results)
        return np.array(results), np.array(real_act)
