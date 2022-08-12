class Stock:
    def __init__(self, name):
        # Stock Id
        self.stock_name = name
        self.time_stamp = 1

        # Momentum
        self.daily_precentile_acceleration = 1
        self.market_volume_acceleration = 1
        self.stock_price_acceleration = 1
        self.shares_availability_acceleration = 1

        # Current
        self.current_price_daily_percentile = 1
        self.last_market_volume = 1
        self.last_stock_price = 1
        self.last_shares_availability = 1

    def update(self, price_per, volume, price, shares_avail, curr_time, lr):
        dlr = lr / (self.time_stamp - curr_time)
        per_var = dlr * (self.current_price_daily_percentile - price_per)
        volume_var = dlr * (self.current_price_daily_percentile - volume)
        price_var = dlr * (self.current_price_daily_percentile - price)
        avail_var = dlr * (self.current_price_daily_percentile - shares_avail)

        self.daily_precentile_acceleration += per_var
        self.market_volume_acceleration += volume_var
        self.stock_price_acceleration += price_var
        self.shares_availability_acceleration += avail_var

        self.current_price_daily_percentile = max(price_per, self.current_price_daily_percentile)
        self.last_market_volume = max(volume, self.last_market_volume)
        self.last_stock_price = max(price, self.last_stock_price, self.last_stock_price)
        self.last_shares_availability = max(shares_avail, self.last_shares_availability)

        return