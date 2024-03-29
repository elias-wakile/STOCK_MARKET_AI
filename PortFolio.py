from Stock import Stock
from StockData import StockData
import numpy as np


class PortFolio:
    def __init__(self, initial_investment, stock_list, interval, start_date, end_date, stock_indices, file,
                 action_space, state_size=7):
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
        self.rewards_dict = {}
        self.file = file
        self.profit = 0
        for stock_name in stock_list:
            self.stocks[stock_name] = Stock(stock_name, self.file)
            self.rewards_dict[stock_name] = []
        self.stock_indices = stock_indices
        self.state_size = state_size
        self.stock_market = {stock_name: StockData(stock_name, start_date,
                                                   self.interval, end_date,
                                                   self.stocks[stock_name], 10e-1)
                             for stock_name in
                             self.stock_name_list}
        self.min_stick_len = min(self.stock_market, key=lambda x: self.stock_market[x].row_len)
        self.action_space = action_space

    def update_portfolio(self):
        """
        This function updates the portfolio for the current date
        """
        for stock_name in self.stock_name_list:
            if not self.stock_market[stock_name].update_stock():
                break

    def getBalance(self):
        """
        Gets the balance of the portfolio
        :return: The sum of balance that we have in cash and the money we could
        have if we immediately sold our stocks.
        """
        stock_balance = 0
        for stock in self.stock_name_list:
            stock_balance += self.stocks[stock].money_in_stock
        print(f"Current balance: {self.balance}."
              f" Stocks' total value: {stock_balance}."
              f" Total value: {stock_balance + self.balance}.")
        self.file.write(f"Current balance: {self.balance}."
                        f" Total stock(s) value: {stock_balance}."
                        f" Total value: {stock_balance + self.balance}.\n")
        return stock_balance + self.balance

    def get_state(self):
        """
        This function gets the current state of the Portfolio
        :return: The state of the Portfolio, each line represents a stock and
        the columns represent the different values of the parameters
        """
        state = np.zeros((len(self.stock_name_list), self.state_size))
        for line_index in self.stock_indices.keys():
            stock_name = self.stock_indices[line_index]
            get_st = self.stock_market[stock_name].get_state()
            state[line_index, :] = get_st

        return state

    def linear_reward(self):
        action = [0] * len(self.stock_name_list)
        reword = [0] * len(self.stock_name_list)
        for index in self.stock_indices.keys():
            stock_name = self.stock_indices[index]
            num_of_stocks = 0
            if self.stocks[stock_name].last_close_price - self.stocks[stock_name].last_open_price > 0:
                num_of_stocks = 1
            elif self.stocks[stock_name].last_close_price - self.stocks[stock_name].last_open_price < 0:
                num_of_stocks = -1

            if num_of_stocks < -self.stocks[stock_name].num_of_stocks_owned:
                num_of_stocks = -self.stocks[stock_name].num_of_stocks_owned
            elif num_of_stocks * self.stocks[stock_name].last_low_price >= self.balance:
                num_of_stocks = int(self.balance / self.stocks[stock_name].last_low_price)
            trade_result = self.stocks[stock_name].transaction(num_of_stocks)
            if num_of_stocks > 0:
                self.balance -= num_of_stocks * self.stocks[stock_name].last_low_price
                self.rewards_dict[stock_name].append(trade_result)
                action[index] = num_of_stocks
            elif num_of_stocks < 0:
                profit = trade_result - self.rewards_dict[stock_name].pop(0)
                reword[index] = profit
                self.profit += profit
                self.balance += -num_of_stocks * self.stocks[stock_name].last_high_price
                action[index] = num_of_stocks
                print(f"Sold {-num_of_stocks} stock(s) of {stock_name}: "
                      f"{self.stocks[stock_name].last_high_price}$ per stock, and make a profit of {profit}$.")
                self.file.write(f"Sold {-num_of_stocks} stock(s) of {stock_name}: "
                                f"{self.stocks[stock_name].last_high_price}$ per stock., and make a profit of {profit}$. \n")
        return action, reword

    def sort_buy(self, stock_predictions):
        but_dic = {}
        for index in stock_predictions:
            stock_name = self.stock_indices[index]
            but_dic[index] = self.stocks[stock_name].last_low_price
        new_dic = {k: v for k, v in sorted(but_dic.items(), key=lambda item: item[1])}
        return new_dic.keys()

    def action(self, stock_predictions):
        """
        This function executes the predictions of the Model
        :param stock_predictions: the number of shares we wanna buy of each stock
                                  the number of shares we wanna buy for AAPL
                                  will be at self.stock_indices["AAPL"]
        :return: A reward vector of the same format
        """
        results = [0] * len(self.stock_name_list)
        real_act = [0] * len(self.stock_name_list)
        action_space_limit = int(self.action_space/2)
        for i in range(1,action_space_limit+1):
            if len(stock_predictions[i]) > 0:
                stock_predictions[i] = self.sort_buy(stock_predictions[i])

        for num_of_stocks in range(-action_space_limit, action_space_limit+1):
            if num_of_stocks == 0:
                continue
            for index in stock_predictions[num_of_stocks]:
                reward = 0
                real_profit = 0
                stock_name = self.stock_indices[index]
                if num_of_stocks < -self.stocks[stock_name].num_of_stocks_owned:
                    num_of_stocks = -self.stocks[stock_name].num_of_stocks_owned
                elif num_of_stocks * self.stocks[stock_name].last_low_price > self.balance:
                    num_of_stocks = int(self.balance / self.stocks[stock_name].last_low_price)

                trade_result = self.stocks[stock_name].transaction(num_of_stocks)
                if trade_result == 0:
                    continue
                if num_of_stocks > 0:
                    reward = 0

                    for j in range(num_of_stocks):
                        self.rewards_dict[stock_name].append(trade_result)
                        self.balance -= self.stocks[stock_name].last_low_price
                elif num_of_stocks < 0:
                    curr_balance = self.balance
                    for j in range(-1*num_of_stocks):
                        profit = trade_result - self.rewards_dict[stock_name].pop(0)
                        real_profit += profit
                        # if profit < 0 :
                        #     reward += profit * 5
                        # else:
                        reward += profit
                        self.profit += profit
                        self.balance += self.stocks[stock_name].last_high_price
                    print(f'Sold {-num_of_stocks} stock(s) of {stock_name}: {self.stocks[stock_name].last_high_price}$'
                          f' per stock, and make a profit of {real_profit}$.')
                    self.file.write(f"Sold {-num_of_stocks} stock(s) of {stock_name}: "
                                    f"{self.stocks[stock_name].last_high_price}$ per stock., and make a profit of"
                                    f" {real_profit}$. \n")
                results[index] = reward

                real_act[index] = num_of_stocks
        return np.array(results)
