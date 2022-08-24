MOMENTUM = 0.1


class Stock:
    """
    This class represents a stock
    """

    def __init__(self, name, file):
        """
        This is a new stock constructor
        :param name: The stock name
        """
        # Stock Id
        self.stock_name = name
        self.time_stamp = 1
        self.daily_lowest = float(0)
        self.daily_highest = float(0)
        self.last_market_volume = float(1)

        # Momentum
        self.daily_precentile_momentum = float(1)
        self.market_volume_momentum = float(1)
        self.stock_price_momentum = float(1)

        # Current
        self.price_var = float(1)
        self.volume_var = float(1)
        self.per_var = float(1)

        # Purchase Data
        self.num_of_stocks_owned = 0
        self.money_in_stock = float(0)
        self.price_per_stock = float(0)

        # Acceleration
        self.market_volume_acceleration = float(1)
        self.daily_precentile_acceleration = float(1)
        self.stock_price_acceleration = float(1)
        self.market_volume_acceleration = float(1)

        self.file = file

        # Price range
        self.current_price_daily_percentile = float(1)
        self.last_high_price = float(1)
        self.last_low_price = float(1)
        self.last_open_price = float(1)
        self.last_close_price = float(1)

        # Special formula parameters
        self.ADX = float(1)
        self.MACD = float(1)
        self.CCI = float(1)
        self.RSI = float(1)

    def update(self, volume, low_price, high_price, open_price,
               close_price, curr_time, RSI, ADX, CCI, MACD, momentum,
               new_day=False, newday_momentum=1e-3):
        """

        :param volume: The current volume of the stock
        :param low_price: The moment lowest price
        :param high_price: The moment highest price
        :param open_price: The moment open price of the stock
        :param close_price: The moment close price of the stock
        :param curr_time: The current time stamp of the day
        :param RSI: The relative strength index of the stock
        :param ADX: The average directional index
        :param CCI: The commodity channel index
        :param MACD: The moving average convergence divergence
        :param momentum: The momentum of the variation
        :param new_day: Is it a new day?
        :param newday_momentum: The newday acceleration momentum
        """
        # Update time
        self.daily_highest = max(self.daily_highest, high_price)
        self.daily_lowest = min(self.daily_lowest, low_price)
        if new_day:
            self.daily_lowest = low_price
            self.daily_highest = high_price
            self.daily_precentile_momentum *= newday_momentum
            self.market_volume_momentum *= newday_momentum
            self.stock_price_momentum *= newday_momentum

        # Compute differentials (momenta)
        per_var = (self.daily_highest - high_price) / self.daily_highest
        volume_var = volume / self.last_market_volume
        price_var = open_price / close_price

        # Update accelerations
        self.time_stamp = curr_time
        if not new_day:
            self.daily_precentile_momentum += momentum * (per_var - self.per_var)
            self.market_volume_momentum += momentum * (volume_var - self.volume_var)
            self.stock_price_momentum += momentum * (price_var - self.price_var)
        if self.per_var != 0:
            self.daily_precentile_acceleration = 1 - per_var / self.per_var
        self.market_volume_acceleration = 1 - (volume_var / self.volume_var)
        self.stock_price_acceleration = 1 - (price_var / self.price_var)

        # Update differentials
        self.per_var = per_var
        self.volume_var = volume_var
        self.price_var = price_var

        # Update dailies
        self.current_price_daily_percentile = close_price / self.daily_highest
        self.last_market_volume = volume
        self.last_high_price = high_price
        self.last_low_price = low_price
        self.last_open_price = open_price
        self.last_close_price = close_price

        # Update special formula parameters
        self.ADX = ADX
        self.MACD = MACD
        self.CCI = CCI
        self.RSI = RSI

        self.money_in_stock = self.num_of_stocks_owned * self.daily_highest

    def buy_stock(self, amount_of_stocks: int):
        """
        This function process a buying of stocks
        :param amount_of_stocks: The amount of stocks to be bought
        :return: The potentiality of this buy
        """
        self.num_of_stocks_owned += amount_of_stocks
        self.money_in_stock += amount_of_stocks * self.last_low_price
        print(f"Bought {amount_of_stocks} stock(s) of {self.stock_name}: "
              f"{self.last_low_price}$ per stock.")
        self.file.write(f"Bought {amount_of_stocks} stock of {self.stock_name}: "
                        f"{self.last_low_price}$ per stock.\n")
        return  self.last_low_price

    def sell_stock(self, amount_of_stocks):
        """
        This function process a selling of stocks
        :param amount_of_stocks: The amount of stocks to be sold
        :return: The potentiality of this sale
        """
        if self.num_of_stocks_owned == 0:
            return 0
        if amount_of_stocks > self.num_of_stocks_owned:
            amount_of_stocks = self.num_of_stocks_owned
        self.num_of_stocks_owned -= amount_of_stocks
        self.money_in_stock -= amount_of_stocks * self.last_high_price
        print(f"Sold {amount_of_stocks} stock(s) of {self.stock_name}: "
              f"{self.last_high_price}$ per stock.")
        self.file.write(f"Sold {amount_of_stocks} stock(s) of {self.stock_name}: "
                        f"{self.last_high_price}$ per stock. \n")
        return self.last_high_price

    def transaction(self, prediction):
        """
        This function proceeds a trade in the stock market
        based on a prediction of the model
        :param prediction:
        :return:
        """
        if prediction < 0:
            return self.sell_stock(prediction * -1)
        elif prediction > 0:
            return self.buy_stock(prediction)
        return 0  # keep is 0
