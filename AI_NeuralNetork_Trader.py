import tensorflow as tf
import datetime
import pandas as pd
from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
import keras.backend as k
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar


def huber_loss(y_true, y_pred, clip_delta=1.0):
    """Huber loss - Custom Loss Function for Q Learning
    Links: 	https://en.wikipedia.org/wiki/Huber_loss
            https://jaromiru.com/2017/05/27/on-using-huber-loss-in-deep-q-learning/
    """
    error = y_true - y_pred
    cond = k.abs(error) <= clip_delta
    squared_loss = 0.5 * k.square(error)
    quadratic_loss = 0.5 * k.square(clip_delta) + clip_delta * (k.abs(error) - clip_delta)
    return k.mean(tf.where(cond, squared_loss, quadratic_loss))


def stocks_price_format(n):
    if n < 0:
        return "- # {0:2f}".format(abs(n))
    else:
        return "$ {0:2f}".format(abs(n))


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


def add_stock_predictions(neural_network, action, states, stock_predictions):
    for ind, name in enumerate(stock_names):
        action.append(
            neural_network.action([states[ind]]) - int(neural_network.action_space / 2))  # make this to be between -1 to 1
        stock_predictions[name] = action[ind]


def run_trader(natural_network, portfolio_obj, batch_len, stocks_names, file):
    i = 0
    done = False
    states = portfolio_obj.get_state().tolist()
    tmp = portfolio_obj.stock_market[portfolio_obj.min_stick_len]

    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_predictions = {}
    for t in tqdm(range(data_samples)):
        action = []
        for ind, name in enumerate(stocks_names):
            action.append(
                natural_network.action([states[ind]]) - int(natural_network.action_space / 2))  # make this to be between -1 to 1
            stock_predictions[name] = action[ind]
        # add_stock_predictions(neuralNet, action, states, stock_predictions)
        portfolio_obj.update_portfolio()
        next_states = portfolio_obj.get_state().tolist()
        results, action_new = portfolio_obj.action(stock_predictions)
        for ind, name in enumerate(stocks_names):
            natural_network.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
        states = next_states
        if len(natural_network.memory) > batch_len:  # todo: I add and t % (batch_size / 2) == 0  mey mastic
            natural_network.batch_train(batch_len)
        i += 1
        portfolio_obj.getBalance()
        print(f'run: {i} from {data_samples}')
        file.write(f'run: {i} from {data_samples}' + '\n')
        if t == data_samples - 1:
            done = True
    portfolio_obj.getBalance()


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
    stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    date_list = make_date_list(5)
    interval = "1m"
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000

    with open("result", 'w') as f:

        porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f)

        # vars for NeuralNetwork
        episodes = 3
        state_size = 7  # todo: if there is more then one stock this need to cheng
        action_space = 3  # todo: if there is more then one stock this need to cheng ODD
        signal_rate = 1  # todo: what is this parmeter?

        neural_net = NeuralNetwork(episodes=episodes, signal_rate=signal_rate, state_size=state_size,
                                   action_space=action_space, load_model='ai_trader_5.h5')

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
                neural_net.model.save("ai_trader_{}.h5".format(episode))
            porfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f)

        print("yes!!!!")
