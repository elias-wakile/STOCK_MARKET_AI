import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar


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
        print(
            f'The date: {portfolio_agent.stock_market[portfolio_agent.stock_name_list[0]].stock_data["DATE"].iloc[t]}')
        file.write(
            f'The date: {portfolio_agent.stock_market[portfolio_agent.stock_name_list[0]].stock_data["DATE"].iloc[t]}' + '\n')
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
        # print(f'run: {i} from {data_samples}')
        # file.write(f'run: {i} from {data_samples}' + '\n')
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
def run_trader_linear(porfolio, file, stock_names):
    i = 0
    tmp = porfolio.stock_market[porfolio.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    balance_difference_lst = []
    current_balance = initial_balance
    balance_progress = []
    balance_difference = initial_balance
    states_buy = [[] for i in stock_names]
    states_sell = [[] for i in stock_names]
    for _ in tqdm(range(data_samples)):
        print(
            f'The date: {porfolio.stock_market[porfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}')
        file.write(
            f'The date: {porfolio.stock_market[porfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}' + '\n')
        actions = porfolio.linear_reward()
        for ind, name in enumerate(stock_names):
            a = actions[ind]
            if a > 0:
                states_buy[ind].append(porfolio.stock_market[name].time_stamp)
            elif a < 0:
                states_sell[ind].append(porfolio.stock_market[name].time_stamp)
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
        # i += 1
        porfolio.getBalance()
        # print(f'run: {i} from {data_samples}')
        # file.write(f'run: {i} from {data_samples} \n')
    balance = porfolio.getBalance()
    fig, ax1 = plt.subplots()
    ax = [ax1]
    for ind, name in enumerate(stock_names):
        close = porfolio.stock_market[stock_names[ind]].stock_data["Close"]

        ax[-1].plot(close, lw=2.)
        ax[-1].plot(close, '^', markersize=10,  markevery=states_buy[ind])
        ax[-1].plot(close, 'v', markersize=10, markevery=states_sell[ind])
        # ax2 = ax[-1].twinx()
        # ax.append(ax2)
    plt.legend()
    invest = ((balance - 10000) / 10000) * 100
    plt.title('total gains %f, total investment %f%%' % (balance, invest))
    plt.savefig('Q-learning agent' + '.png')
    plt.show()
    print(porfolio.profit)



def main_def(kind_agent, is_new_net):
    # vars for PortFolio
    # stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    # date_list = make_date_list(5)
    # interval = "1m"
    # stock_names = ["AMD", "BABA", "SIGA", "CCI", "KPRX"]
    # stock_names = ["DELL", "T", "VZ", "IBM", "BABA"]
    stock_names = ["AAPL", "GOOGL", "MSFT", "META", "AMZN"]

    date_list = make_date_list(200)
    interval = "1d"
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000
    haf_len_date = int(len(date_list) / 2)
    # vars for NeuralNetwork
    episodes = 10
    state_size = 7
    action_space = 3
    graph_index = 1

    with open("testing_interface.txt", 'w') as f:
        portfolio = PortFolio(initial_investment, stock_names, interval, date_list[:haf_len_date], stock_indices, f,
                              action_space)
        if is_new_net == "1":
            neural_net = NeuralNetwork(episodes=episodes, state_size=state_size,
                                       action_space=action_space, model_to_load='5_HI_TECH_trader_100.h5')
        else:
            neural_net = NeuralNetwork(episodes=episodes, state_size=state_size, action_space=action_space)
        batch_size: int = 16
        if kind_agent == "1" or kind_agent == "3":

            for episode in range(1, episodes + 1):
                print("Episode: {}/{}".format(episode, episodes))
                f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                f.write("Episode: {}/{}".format(episode, episodes) + '\n')
                f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
                run_trader(neural_net, portfolio, batch_size, stock_names, f)

                if episode % 5 == 0:
                    neural_net.model.save("testing_interface_{}.h5".format(episode))
                portfolio = PortFolio(initial_investment, stock_names, interval, date_list[:haf_len_date],
                                      stock_indices, f,
                                      action_space)
            portfolio = PortFolio(initial_investment, stock_names, interval, date_list[haf_len_date:], stock_indices, f,
                                  action_space)
            print("Test")
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Test" + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            run_trader(neural_net, portfolio, batch_size, stock_names, f)
        if kind_agent == "2" or kind_agent == "3":
            print("Test")
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Test" + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            run_trader_linear(portfolio, f, stock_names)


if __name__ == "__main__":
    agent_kind = input('Press 1 for run Neural Net agent and 2 for Extrapolation agent or 3 for run both agents\n')
    new_net = None
    if agent_kind == "1" or agent_kind == "3":
        new_net = input("Press 1 for lode existing neural net and 2 for run on new one\n")
    main_def(agent_kind, new_net)
