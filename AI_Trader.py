import math
import random
import datetime
import pandas_ta as pta
import numpy as np
import pandas as pd
import tensorflow as tf
#import pandas_datareader as data_reader
# import talib
# from ta.trend import ADXIndicator
from sklearn.model_selection import train_test_split
from keras.models import load_model, clone_model

import keras.backend as K
from tqdm import tqdm_notebook, tqdm
from collections import deque
import yfinance as yf
import csv


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


class AI_Trader:
    def __init__(self, state_size, action_space=3, model_name="AITrader", balance=10000, lodModel=None, epsilon=1.0):

        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name
        self.loss = huber_loss

        self.gamma = 0.95
        self.epsilon = epsilon
        self.epsilon_final = 0.01
        self.epsilon_decay = 0.995
        self.balance = balance
        self.num_own_stock = 0
        self.money_in_stock = 0
        self.avg_stock_p = 0
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        if lodModel is None:
            self.model = self.model_builder()
        else:
            self.model = self.load(lodModel)

    def load(self, model_name):
        return load_model(model_name, custom_objects=self.custom_objects)

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def trade(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions)

    def reset_episod(self, balance):
        self.balance = balance
        self.num_own_stock = 0
        self.money_in_stock = 0
        self.avg_stock_p = 0

    def batch_train(self, batch_size):

        # batch = random.sample(self.memory, batch_size)
        batch = []
        for i in range(len(self.memory) - batch_size + 1, len(self.memory)):
            batch.append(self.memory[i])

        X_train, y_train = [], []

        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            q_values = self.model.predict(state)
            # update the target for current action based on discounted reward
            q_values[0][action] = target
            X_train.append(state[0])
            y_train.append(q_values[0])

        loss = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0).history["loss"][0]

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        return loss


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def stocks_price_format(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


def dataset_loader(stock_name, delta_days, interval):
    tic = yf.Ticker(stock_name)
    start_date = datetime.datetime.now() - datetime.timedelta(delta_days)
    end_date = (datetime.datetime.now() - datetime.timedelta(1))
    dataset = tic.history(start=start_date.strftime('%Y-%m-%d'), interval=interval, end=end_date.strftime('%Y-%m-%d'))
    dataset["RSI"] = pta.rsi(dataset["Close"], length=14)
    dataset["ADX"] = pta.adx(dataset["High"], dataset["Low"], dataset["Close"], length=7)["ADX_7"]
    dataset["MACD"] = pta.macd(dataset["Close"], fast=4, slow=12, signal=3)["MACDs_4_12_3"]
    dataset["CCI"] = pta.cci(dataset["High"], dataset["Low"], dataset["Close"], length=14)
    return dataset[14:]


def get_state(data, t, n_days):
    """Returns an n-day state representation ending at time t
    """
    # d = t - n_days + 1
    # block = data[d: t + 1] if d >= 0 else (-d * [data[0]]) + data[0: t + 1]  # pad with t0
    # res = []
    # for i in range(n_days - 1):
    #     res.append(sigmoid(block[i + 1] - block[i]))
    # return np.array([res])
    starting_id = t - n_days + 1
    if starting_id >= 0:
        windowed_data = data[starting_id:t + 1]
    else:
        windowed_data = abs(starting_id) * [data[0]] + list(data[0:t + 1])
    state = []
    for i in range(n_days - 1):
        state.append(sigmoid(windowed_data[i + 1] - windowed_data[i]))

    return np.array([state])


def state_creator(dataset, timestep, trader):
    if timestep == 0:
        state = [0.5] * 5
        state.append(1)
        state.append(0)
        return [state]
    data_c = dataset.iloc[timestep]
    data_p = dataset.iloc[timestep - 1]
    state = [sigmoid(data_c["Close"] - data_p["Close"]), sigmoid(data_c["MACD"] - data_p["MACD"]),
             sigmoid(data_c["RSI"] - data_p["RSI"]), sigmoid(data_c["CCI"] - data_p["CCI"]),
             sigmoid(data_c["ADX"] - data_p["ADX"])]
    if data_c['Close'] > trader.balance:
        state.append(0)
    else:
        state.append(1)

    if trader.num_own_stock > 0:
        state.append(1)
    else:
        state.append(0)
    return np.array([state])


def run_trader(trader, data, batch_size, is_train=True):
    trader.memory.clear()  # todo: I add this
    data_samples = data.shape[0] - 1
    # state = state_creator(data, 0, trader)
    d = [data["Close"]]
    # state = get_state(d[0].values, 0, trader.state_size + 1)
    state = state_creator(data, 0, trader)
    total_profit = 0
    trader.inventory = []
    done = False

    act_chose = {0: 0, 1: 0, 2: 0}
    act_do = {0: 0, 1: 0, 2: 0}

    # for t in tqdm(range(data_samples)):
    for t in range(data_samples):

        action = trader.transaction(state)
        # next_state = get_state(d[0].values, t + 1, trader.state_size + 1)
        next_state = state_creator(data, t + 1, trader)
        reward = 0
        if action == 1:  # Buying
            act_chose[1] += 1
            if trader.balance >= data['Close'][t]:
                act_do[1] += 1
                trader.inventory.append(data['Close'][t])
                trader.balance -= data['Close'][t]
                trader.avg_stock_p = (trader.avg_stock_p * trader.num_own_stock + data['Close'][t]) / (
                        trader.num_own_stock + 1)
                trader.num_own_stock += 1
                trader.money_in_stock += data['Close'][t]  # todo: may update every iteration
                print("AI Trader bought: ", stocks_price_format(data['Close'][t]))
            else:
                act_do[0] += 1
                # print("!!!AI Trader want to buy but can't do this!!!")
        elif action == 2:  # Selling
            act_chose[2] += 1
            if len(trader.inventory) > 0:
                act_do[2] += 1
                buy_price = trader.inventory.pop(0)
                trader.balance += data['Close'][t]
                trader.num_own_stock -= 1
                trader.money_in_stock -= trader.avg_stock_p
                reward = data['Close'][t] - buy_price  # todo: may need her max with 0
                total_profit += data['Close'][t] - trader.avg_stock_p
                print("AI Trader sold: ", stocks_price_format(data['Close'][t]),
                      " Profit: " + stocks_price_format(data['Close'][t] - buy_price))
            else:
                act_do[0] += 1
                # print("!!!AI Trader want to sold but can't do this!!!")
        else:
            # print("AI Trader skip")
            act_chose[0] += 1
            act_do[0] += 1

        if t == data_samples - 1:
            done = True

        trader.memory.append((state, action, reward, next_state, done))

        state = next_state
        val_own_stock = trader.num_own_stock * data['Close'][t]

        if t % 50 == 0:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("Balance: {}".format(trader.balance))
            print("Num Stock: {}".format(trader.num_own_stock))
            print("VAL Of own stock: {}".format(val_own_stock))
            print("TOTAL VAL: {}".format(trader.balance + val_own_stock))
            print(f"Chose 0: {act_chose[0]} times and do {act_do[0]} times")
            print(f"Chose 1: {act_chose[1]} times and do {act_do[1]} times")
            print(f"Chose 2: {act_chose[2]} times and do {act_do[2]} times")
            print("########################")

        if done:
            print("########################")
            print("TOTAL PROFIT: {}".format(total_profit))
            print("TOTAL VAL: {}".format(trader.balance + val_own_stock))
            print(f"Chose 0: {act_chose[0]} times and do {act_do[0]} times")
            print(f"Chose 1: {act_chose[1]} times and do {act_do[1]} times")
            print(f"Chose 2: {act_chose[2]} times and do {act_do[2]} times")
            print("########################")

        if len(trader.memory) > batch_size:
            if not is_train and t % (batch_size / 2) == 0:
                trader.batch_train(batch_size)
            else:
                trader.batch_train(batch_size)


if __name__ == "__main__":

    stock_list = ["NDAQ", "GOOGL", "AAPL", "NVDA", "AXSM"]
    for stock_name in stock_list:
        print(f"#####  start stock {stock_name} #######")
        data = dataset_loader(stock_name, 4, "1m")
        name_csv = stock_name + "_oneDay_test.csv"
        data.to_csv(name_csv)
        state_size = 7  # 4
        episodes = 5
        train, test = train_test_split(data, test_size=0.5, shuffle=False)
        budget = 10000
        batch_size = 16  # 64
        data_samples = train.shape[0] - 1
        trader = AI_Trader(state_size, lodModel="test.h5")
        for episode in range(1, episodes + 1):
            shares = 0
            print("Episode: {}/{}".format(episode, episodes))
            # state = state_creator(data, 0, window_size + 1)
            run_trader(trader, train, batch_size)
            trader.reset_episod(budget)

            if episode % 5 == 0:
                trader.model.save("test.h5")
                # trader.model.save(stock_name + "_6_AI_trader_{}.h5".format(episode))
        print("#####  start test  #######")
        run_trader(trader, test, batch_size)
