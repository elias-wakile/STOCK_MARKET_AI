import tensorflow as tf
from pandas.tseries.offsets import BDay
import datetime
import pandas as pd

from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
import keras.backend as K
from tqdm import tqdm
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
    states = porfolio.get_state().tolist()
    tmp = porfolio.stock_market[porfolio.min_stick_len]

    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_predictions = {}
    action_limit = int(neuralNet.action_space / 2)
    for t in tqdm(range(data_samples)):
        action = []
        # action_dic = {-1: [], 0: [], 1: []}
        action_dic = {i: [] for i in range(-action_limit, action_limit + 1)}
        for ind, name in enumerate(stock_names):
            a = neuralNet.action([states[ind]]) - int(neuralNet.action_space / 2)
            action.append(a)  # make this to be between -1 to 1
            stock_predictions[name] = action[ind]
            action_dic[a].append(ind)
        # add_stock_predictions(neuralNet, action, states, stock_predictions)
        porfolio.update_portfolio()
        next_states = porfolio.get_state().tolist()
        results = porfolio.action(action_dic)
        for ind, name in enumerate(stock_names):
            neuralNet.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
        states = next_states
        porfolio.getBalance()
        if len(neuralNet.memory) > batch_size:
            neuralNet.batch_train(batch_size)
        i += 1
        print(f'run: {i} from {data_samples}')
        file.write(f'run: {i} from {data_samples}' + '\n')
        if t == data_samples - 1:
            done = True
    porfolio.getBalance()


def run_trader_linear(porfolio, file):
    i = 0
    tmp = porfolio.stock_market[porfolio.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    for _ in tqdm(range(data_samples)):
        action = []
        action.append(porfolio.linear_reward())
        porfolio.update_portfolio()
        i += 1
        porfolio.getBalance()
        print(f'run: {i} from {data_samples}')
        file.write(f'run: {i} from {data_samples} \n')
    porfolio.getBalance()


if __name__ == "__main__":
    # vars for PortFolio
    # stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    stock_names = ["AAPL"]
    date_list = make_date_list(380)
    interval = "1d"
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000
    # vars for NeuralNetwork
    episodes = 2
    state_size = 7
    action_space = 7
    with open("result", 'w') as f:

        neural_net = NeuralNetwork(episodes=episodes, state_size=state_size,
                                   action_space=action_space)#, model_to_load='ai_trader_5.h5')
        porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f, action_space)

        batch_size: int = 16

        for episode in range(1, episodes + 1):
            shares = 0
            print("Episode: {}/{}".format(episode, episodes))
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Episode: {}/{}".format(episode, episodes) + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            run_trader(neural_net, porfolio, batch_size, stock_names, f)
            # run_trader_linear(porfolio, f)
            if episode % 5 == 0:
                neural_net.model.save("ai_1_trader_{}.h5".format(episode))
            porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f, action_space)
