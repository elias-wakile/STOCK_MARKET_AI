import datetime

import numpy as np
import pandas as pd
from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
import matplotlib.pyplot as plt

def make_date_list(delta_days):
    # we choose the 24th of August 2022 to have uniformed graphs
    start_date = datetime.date(2022,8,24) - datetime.timedelta(delta_days)
    end_date = datetime.date(2022,8,24)
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


def run_trader(neuralNet, portfolio_agent, batch_size, stock_names, file,initial_balance,graph_index):
    i = 0
    done = False
    states = portfolio_agent.get_state().tolist()
    tmp = portfolio_agent.stock_market[portfolio_agent.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_predictions = {}
    balance_difference_lst = []
    current_balance = initial_balance
    balance_difference = initial_balance
    balance_progress = []
    action_limit = int(neuralNet.action_space / 2)
    for t in tqdm(range(data_samples)):
        action = []
        action_dic = {i: [] for i in range(-action_limit, action_limit + 1)}
        for ind, name in enumerate(stock_names):
            a = neuralNet.action([states[ind]]) - int(neuralNet.action_space / 2)  # make this to be between -X to X
            action.append(a)
            stock_predictions[name] = action[ind]
            action_dic[a].append(ind)
        portfolio_agent.update_portfolio()
        next_states = portfolio_agent.get_state().tolist()
        results = portfolio_agent.action(action_dic)
        for ind, name in enumerate(stock_names):
            neuralNet.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
        states = next_states
        balance_progress.append(current_balance)
        current_balance = portfolio_agent.getBalance()
        balance_difference -= current_balance
        balance_difference_lst.append(balance_difference)
        balance_difference = current_balance

        if len(neuralNet.memory) > batch_size:
            neuralNet.batch_train(batch_size)
        i += 1
        print(f'run: {i} from {data_samples}')
        file.write(f'run: {i} from {data_samples}' + '\n')
        if t == data_samples - 1:
            done = True
    # balance_difference = current_balance
    # current_balance = portfolio_agent.getBalance()
    # balance_difference -= current_balance
    # balance_difference_lst.append(balance_difference)
    xaxis = np.array([portfolio_agent.date_list[i] for i in range(data_samples)])
    yaxis = np.array(balance_difference_lst)
    plt.figure(figsize=(15, 5))
    plt.plot(xaxis,yaxis)
    plt.savefig("reward_graph_{}.1.png".format(graph_index))
    plt.close()

    y1axis = np.array(balance_progress)
    plt.figure(figsize=(15, 5))
    plt.plot(xaxis, y1axis)
    plt.savefig("balance_progress_{}.1.png".format(graph_index))
    plt.close()


def run_trader_linear(porfolio, file,initial_balance):
    i = 0
    tmp = porfolio.stock_market[porfolio.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    balance_difference_lst = []
    current_balance = initial_balance
    balance_progress = []
    balance_difference = initial_balance
    for _ in tqdm(range(data_samples)):
        action = []
        action.append(porfolio.linear_reward())
        porfolio.update_portfolio()
        i += 1
        # porfolio.getBalance()
        balance_progress.append(current_balance)
        current_balance = porfolio.getBalance()
        balance_difference -= current_balance
        balance_difference_lst.append(balance_difference)
        balance_difference = current_balance

        print(f'run: {i} from {data_samples}')
        file.write(f'run: {i} from {data_samples} \n')
    # porfolio.getBalance()
    xaxis = np.array([porfolio.date_list[i] for i in range(data_samples)])
    yaxis = np.array(balance_difference_lst)
    plt.plot(xaxis, yaxis)
    plt.xticks(rotation=45)
    plt.savefig("reward_graph.png")
    plt.close()

    y1axis = np.array(balance_progress)
    plt.plot(xaxis, y1axis)
    plt.xticks(rotation=45)
    plt.savefig("balance_progress.png")
    plt.close()


if __name__ == "__main__":
    # vars for PortFolio
    stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    date_list = make_date_list(80)
    interval = "1d"
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000
    # vars for NeuralNetwork
    episodes = 4
    state_size = 7
    action_space = 3
    graph_index = 1
    with open("result", 'w') as f:

        neural_net = NeuralNetwork(episodes=episodes, state_size=state_size,
                                   action_space=action_space, model_to_load='ai_trader_5.h5')
        portfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f, action_space)

        batch_size: int = 16

        for episode in range(1, episodes + 1):
            shares = 0
            print("Episode: {}/{}".format(episode, episodes))
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Episode: {}/{}".format(episode, episodes) + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            # run_trader(neural_net, portfolio, batch_size, stock_names, f,initial_investment,graph_index)
            run_trader_linear(portfolio, f,initial_investment)
            if episode % 5 == 0:
                neural_net.model.save("ai_1_trader_{}.h5".format(episode))
            portfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f, action_space)
            graph_index+=1