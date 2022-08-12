from Stock import Stock
import pandas_datareader.data as pdr

class PortFolio:
    def __init__(self, initial_investment, stock_list):
        self.stock_name_list = stock_list
        self.stocks = dict()
        self.balance = initial_investment
        for stock_name in stock_list:
            self.stocks[stock_name] = Stock(stock_name)

    def update_portfolio(self, date):
        for stock_name in self.stock_name_list:
            self.stocks[stock_name].update()

    def getBalance(self):
        stock_balance = 0
        for stock in self.stock_list:
            stock_balance += stock.money_in_stock
        print(f"Your Portfolio currently has a balance of {self.balance}, owns"
              f"a stock of total value {stock_balance} and is of total value of"
              f"{stock_balance + self.balance}.")
        return stock_balance + stock

