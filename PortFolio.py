from Stock import Stock
from StockMarket import StockData
import pandas_datareader.data as pdr

class PortFolio:
    def __init__(self, initial_investment, stock_list, period, date_list):
        """
        This function is a portfolio constructor
        :param initial_investment: The initial investment in the portfolio
        :param stock_list: the list of stocks we wish to invest in
        """
        self.stock_name_list = stock_list
        self.stocks = dict()
        self.balance = initial_investment
        self.period = period
        for stock_name in stock_list:
             self.stocks[stock_name] = Stock(stock_name)
        self.date_list = date_list


    def next_day(self):
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
        :return:
        """
        stock_balance = 0
        for stock in self.stock_name_list:
            stock_balance += self.stocks[stock].money_in_stock
        print(f"Your Portfolio currently has a balance of {self.balance}, owns"
              f"a stock of total value {stock_balance} and is of total value of"
              f"{stock_balance + self.balance}.")
        return stock_balance + self.balance

    def act(self, stock_predictions):
        results = {}
        for stock_name in stock_predictions.keys():
            num_of_stocks = stock_predictions[stock_name]
            if num_of_stocks < -self.stocks[stock_name].num_of_stocks:
                num_of_stocks = self.stocks[stock_name].num_of_stocks
            elif num_of_stocks * self.stocks[stock_name].last_low_price >= self.balance:
                num_of_stocks = int(self.balance / (self.stocks[stock_name].last_low_price))
            curr_loss = stock.trade()
            if num_of_stocks < 0:
                self.balance += num_of_stocks * self.stocks[stock_name].last_high_price
            elif num_of_stocks > 0:
                self.balance -= num_of_stocks * self.stocks[stock_name].last_low_price
            results[stock_name] = curr_loss
        return results