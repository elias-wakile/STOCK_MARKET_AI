from Stock import Stock
from StockMarket import StockData
import pandas_datareader.data as pdr

class PortFolio:
    def __init__(self, initial_investment, stock_list, period, interval, date_list):
        """
        This function is a portfolio constructor
        :param initial_investment: The initial investment in the portfolio
        :param stock_list: the list of stocks we wish to invest in
        """
        self.stock_name_list = stock_list
        self.stocks = dict()
        self.balance = initial_investment
        self.period = period
        self.interval = interval
        for stock_name in stock_list:
             self.stocks[stock_name] = Stock(stock_name)
        self.date_list = date_list


    def next_day(self):
        curr_date = self.date_list[0]
        self.date_list = self.date_list[1:]
        self.stock_market = {stock_name : StockData(stock_name, curr_date,
                                                    self.period,self.interval,
                                                    self.stocks[stock_name], 10e-1)
                             for stock_name in self.stock_name_list}

    def update_portfolio(self):
        """
        This function updates the portfolio for the current date
        :param date:
        :return:
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

