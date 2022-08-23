import tensorflow as tf
from pandas.tseries.offsets import BDay
import datetime
import pandas as pd

from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
import keras.backend as K
from tqdm import tqdm_notebook, tqdm
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


def stocks_price_format(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


# def state_creator(dataset, timestep):
#     data = dataset.iloc[timestep]
#     state = [data["Adj Close"]]
#     state.append(data["MACD"])
#     state.append(data["RSI"])
#     state.append(data["CCI"])
#     state.append(data["ADX"])
#     return np.array([state])
#
#
# def run_trader(trader, data, batch_size):
#     # trader.memory.clear()  # todo: I add this
#     data_samples = data.shape[0] - 1
#     state = state_creator(data, 0)
#     total_profit = 0
#     trader.inventory = []
#     done = False
#
#     for t in tqdm(range(data_samples)):
#
#         action = trader.trade(state)
#         next_state = state_creator(data, t + 1)
#         reward = 0
#         if action == 1:  # Buying
#             trader.inventory.append(data['Close'][t])
#             print("AI Trader bought: ", stocks_price_format(data['Close'][t]))
#         elif action == 2 and len(trader.inventory) > 0:  # Selling
#             buy_price = trader.inventory.pop(0)
#
#             reward = data['Close'][t] - buy_price  # todo: may need her max with 0
#             total_profit += data['Close'][t] - buy_price
#             print("AI Trader sold: ", stocks_price_format(data['Close'][t]),
#                   " Profit: " + stocks_price_format(data['Close'][t] - buy_price))
#
#         if t == data_samples - 1:  # todo :
#             done = True
#
#         trader.memory.append((state, action, reward, next_state, done))
#
#         state = next_state
#
#         if done:
#             print("########################")
#             print("TOTAL PROFIT: {}".format(total_profit))
#             print("########################")
#
#         if len(trader.memory) > batch_size:  # todo: I add and t % (batch_size / 2) == 0  mey mastic
#             trader.batch_train(batch_size)


def make_date_list(delta_days):
    start_date = datetime.datetime.now() - datetime.timedelta(delta_days)
    end_date = (datetime.datetime.now() - datetime.timedelta(1))
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d')).to_pydatetime()
    holidays_f = []
    for day in holidays:
        holidays_f.append(day.strftime('%Y-%m-%d'))
    date_ls = pd.date_range(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), freq="B")
    date_list_f = []
    for day_w in date_ls:
        tmp = day_w.strftime('%Y-%m-%d')
        if tmp not in holidays_f:
            date_list_f.append(tmp)

    return date_list_f


def run_trader(neuralNet, porfolio, batch_size, stock_names, file):
    i = 0
    done = False
    states = porfolio.getState().tolist()
    size_loop = porfolio.stock_market[stock_names[0]].stock_data.shape[0] * len(date_list)
    tmp = porfolio.stock_market[porfolio.stock_name_list[0]]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_predictions = {}
    for t in range(data_samples):
        action = []
        for ind, name in enumerate(stock_names):
            action.append(
                neuralNet.act([states[ind]]) - int(neuralNet.action_space / 2))  # make this to be between -1 to 1
            stock_predictions[name] = action[ind]
        porfolio.update_portfolio()
        next_states = porfolio.getState().tolist()
        results, action_new = porfolio.act(stock_predictions)
        for ind, name in enumerate(stock_names):
            neuralNet.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
        states = next_states
        if len(neuralNet.memory) > batch_size:  # todo: I add and t % (batch_size / 2) == 0  mey mastic
            neuralNet.batch_train(batch_size)
        i += 1
        porfolio.getBalance()
        # if i % 25 == 0:
        print(f'run:{i} from {size_loop}')
        file.write(f'run:{i} from {size_loop}')
        if t == data_samples - 1:
            done = True
    porfolio.getBalance()


if __name__ == "__main__":
    # vars for PortFolio
    stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    date_list = make_date_list(5)
    interval = "1m"
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000

    with open("result", 'w') as f:

        porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices,f)

        # vars for NeuralNetwork
        episodes = 3
        state_size = 7  # todo: if there is more then one stock this need to cheng
        action_space = 3  # todo: if there is more then one stock this need to cheng ODD
        signal_rate = 1  # todo: what is this parmeter?

        neuralNet = NeuralNetwork(episodes, signal_rate, stock_names, state_size, action_space, lodModel="ai_trader_5.h5")

        batch_size: int = 16

        for episode in range(1, episodes + 1):
            shares = 0
            print("Episode: {}/{}".format(episode, episodes))
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Episode: {}/{}".format(episode, episodes) + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            run_trader(neuralNet, porfolio, batch_size, stock_names, f)

            if episode % 5 == 0:
                neuralNet.model.save("ai_trader_{}.h5".format(episode))
            porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f)

        print("yes!!!!")
