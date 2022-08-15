import math
import numpy as np
import tensorflow as tf
from pandas.tseries.offsets import BDay
import datetime
import pandas as pd
from PortFolio import PortFolio
import keras.backend as K
from tqdm import tqdm_notebook, tqdm
from collections import deque
from pandas.tseries.holiday import USFederalHolidayCalendar

def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = K.abs(error) <= clip_delta
    squared_loss = 0.5 * K.square(error)
    quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)
    return K.mean(tf.where(cond, squared_loss, quadratic_loss))



def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))




def state_creator(dataset, timestep):
    data = dataset.iloc[timestep]
    state = [data["Adj Close"]]
    state.append(data["MACD"])
    state.append(data["RSI"])
    state.append(data["CCI"])
    state.append(data["ADX"])
    return np.array([state])


def run_trader(trader, data, batch_size):
    # trader.memory.clear()  # todo: I add this
    data_samples = data.shape[0] - 1
    state = state_creator(data, 0)
    total_profit = 0
    trader.inventory = []
    done = False

    for t in tqdm(range(data_samples)):

        action = trader.trade(state)
        next_state = state_creator(data, t + 1)
        reward = 0
        if action == 1:  # Buying
            trader.inventory.append(data['Close'][t])
            print("AI Trader bought: ", stocks_price_format(data['Close'][t]))
        elif action == 2 and len(trader.inventory) > 0:  # Selling
            buy_price = trader.inventory.pop(0)

            reward = data['Close'][t] - buy_price  # todo: may need her max with 0
            total_profit += data['Close'][t] - buy_price
            print("AI Trader sold: ", stocks_price_format(data['Close'][t]),
                  " Profit: " + stocks_price_format(data['Close'][t] - buy_price))

        if t == data_samples - 1: # todo :
            done = True

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("########################")

        if len(trader.memory) > batch_size:  # todo: I add and t % (batch_size / 2) == 0  mey mastic
            trader.batch_train(batch_size)
def make_date_list():
    start_date = datetime.datetime.now() - datetime.timedelta(45)
    end_date = (datetime.datetime.now() - datetime.timedelta(1))
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')).to_pydatetime()
    holidays_f = []
    for day in holidays:
        holidays_f.append(day.strftime('%Y-%m-%d'))
    date_list = pd.date_range(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), freq="B")
    date_list_f = []
    for day_w in date_list:
        tmp = day_w.strftime('%Y-%m-%d')
        if tmp not in holidays_f:
            date_list_f.append(tmp)

    return date_list_f


if __name__ == "__main__":
    stock_name = ["AAPL"]
    date_list = make_date_list()

    porfolio = PortFolio(1000, stock_name, "2m", date_list, {name: i for name , i in enumerate(stock_name)})
    i = 0
    while not porfolio.end_data():
        b = porfolio.getState()
        porfolio.update_portfolio()
        print(i)
        i += 1
    print("yes!!!!")


    a = 7


