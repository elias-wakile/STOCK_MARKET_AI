import datetime
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from NeuralNetwork import NeuralNetwork
from PortFolio import PortFolio
from tqdm import tqdm
from pandas.tseries.holiday import USFederalHolidayCalendar
import plotly.graph_objects as go
import os
from plotly.subplots import make_subplots


def make_date_list(delta_days):
    # we choose the 24th of August 2022 to have uniformed graphs
    start_date = datetime.date(2022, 8, 24) - datetime.timedelta(delta_days)
    end_date = datetime.date(2022, 8, 24)
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


# def buy_sell_graph(model_name,porfolio, stock_names, states_buy, states_sell, graph_index, interval):
#     if graph_index != -1:
#         title = f'Buy and Sell stocks with interval {interval} ' + model_name + \
#                 f" as depend in time: training number {graph_index}"
#         save_name = model_name + f"_buy&sell_stocks_{graph_index}_{interval}.png"
#     else:
#         title = f'Buy and Sell stocks with interval {interval} ' + model_name + f" as depend in time: testing"
#         save_name = model_name + f"_buy&sell_stocks_{interval}.png"
#     fig = make_subplots(rows=len(stock_names), shared_xaxes=True, y_title="Money in dollars", x_title="Date")
#     for ind, name in enumerate(stock_names):
#         fig.add_trace(go.Scatter(x=porfolio.stock_market[stock_names[ind]].stock_data.index,
#                                  y=porfolio.stock_market[stock_names[ind]].stock_data["Close"],
#                                  mode='lines', name=name), row=ind + 1, col=1)
#         fig.add_trace(go.Scatter(x=porfolio.stock_market[stock_names[ind]].stock_data.index[states_buy[ind]], y=porfolio.stock_market[stock_names[ind]].stock_data["Close"].index[states_buy[ind]],
#                                  mode='markers', name=name), row=ind + 1, col=1)
#     fig.update_layout(title_text=title, title_x=0.5)
#     fig.write_image(f"images/{save_name}")
#     fig.show()
#     balance = porfolio.getBalance()
    # fig, ax1 = plt.subplots(len(stock_names), 1)
    # ax = [ax1]
    # fig, ax = plt.subplots(nrows=len(stock_names), ncols=1, sharex="col")
    # for ind, name in enumerate(stock_names):
    #     close = porfolio.stock_market[stock_names[ind]].stock_data["Close"]
    #     # plt.subplot(len(stock_names), 1, ind + 1)
    #     ax[ind].plot(close, lw=2.)
    #     ax[ind].plot(close, '^', markersize=10, markevery=states_buy[ind])
    #     ax[ind].plot(close, 'v', markersize=10, markevery=states_sell[ind])
    # plt.legend()
    # invest = ((balance - 10000) / 10000) * 100
    # plt.title('total gains %f, total investment %f%%' % (balance, invest))
    # plt.savefig('Q-learning agent' + '.png')
    # plt.show()
    # print(porfolio.profit)


def reward_graph(model_name, ax, ay, graph_index, interval):
    fig = go.Figure(data=go.Scatter(x=ax, y=ay, mode='lines'))
    fig.update_yaxes(title_text="Reward in dollars ")
    fig.update_xaxes(title_text="Date")
    if graph_index != -1:
        title = f'Reward of the model with interval {interval} ' + model_name + \
                f" as depend in time: training number {graph_index}"
        save_name = model_name + f"_1reward_graph_{graph_index}_{interval}.png"
    else:
        title = f'Reward of the model with interval {interval} ' + model_name + f" as depend in time: testing"
        save_name = model_name + f"_reward_graph_test_{interval}.png"
    fig.update_layout(title_text=title, title_x=0.5)
    fig.write_image(f"images/{save_name}")
    fig.show()
    plt.plot(ax, ay)


def balance_graph(model_name, ax, ay, graph_index, interval):
    fig = go.Figure(data=go.Scatter(x=ax, y=ay, mode='lines'))
    fig.update_yaxes(title_text="Money in dollars ")
    fig.update_xaxes(title_text="Date")
    if graph_index != -1:
        title = f'Balance of the model with interval {interval} ' + model_name + \
                f" as depend in time: training number {graph_index}"
        save_name = model_name + f"_balance_progress_{graph_index}_{interval}.png"
    else:
        title = f'Balance of the model with interval {interval} ' + model_name + f" as depend in time: testing"
        save_name = model_name + f"_1balance_progress_test_{interval}.png"
    fig.update_layout(title_text=title, title_x=0.5)
    fig.write_image(f"images/{save_name}")
    fig.show()


def balance_graph_together(ax, ay_net,ay_ex, interval):
    fig = go.Figure()
    ax = [i for i in range(len(ay_ex))]
    fig.add_trace(go.Scatter(x=ax, y=ay_net, mode='lines', name="neuralNet"))
    fig.add_trace(go.Scatter(x=ax, y=ay_ex, mode='lines', name="Extrapolation"))
    fig.update_yaxes(title_text="Money in dollars ")
    fig.update_xaxes(title_text="Date")
    title = f'Balance of the models with interval {interval} ' + f" as depend in time"
    save_name = f"balance_progress_two_models_{interval}.png"
    fig.update_layout(title_text=title, title_x=0.5)
    fig.write_image(f"images/{save_name}")
    fig.show()


def reward_pre_stock_graph(model_name, xaxis, stock_reward, graph_index, stock_names, interval):
    if graph_index != -1:
        title = f'Reward per stock with interval {interval} ' + model_name + \
                f" as depend in time: training number {graph_index}"
        save_name = model_name + f"_1reward_per_stock_progress_{graph_index}_{interval}.png"
    else:
        title = f'Reward per stock with interval {interval} ' + model_name + f" as depend in time: testing"
        save_name = model_name + f"_reward_per_stock_progress_test_{interval}.png"
    fig = make_subplots(rows=len(stock_names), shared_xaxes=True, y_title="Money in dollars", x_title="Date")
    for ind, name in enumerate(stock_names):
        fig.add_trace(go.Scatter(x=xaxis, y=stock_reward[name], mode='markers', name=name), row=ind + 1, col=1)
    fig.update_layout(title_text=title, title_x=0.5)
    fig.write_image(f"images/{save_name}")
    fig.show()


def run_trader(neuralNet, portfolio_agent, batch_size, stock_names, file, initial_balance, graph_index, interval):
    i = 0
    done = False
    states = portfolio_agent.get_state().tolist()
    tmp = portfolio_agent.stock_market[portfolio_agent.min_stick_len]
    start_date_ind = tmp.time_stamp
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    stock_reward = {name: [] for name in stock_names}
    balance_difference_lst = []
    current_balance = initial_balance
    balance_difference = 0
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
            # stock_predictions[name] = action[ind]
            action_dic[a].append(ind)
        portfolio_agent.update_portfolio()
        next_states = portfolio_agent.get_state().tolist()
        results = portfolio_agent.action(action_dic)
        for ind, name in enumerate(stock_names):
            neuralNet.memory.append(([states[ind]], action[ind], results[ind], [next_states[ind]], done))
            if results[ind] > 0:
                stock_reward[name].append(results[ind])
            else:
                stock_reward[name].append(results[ind])
        states = next_states
        balance_progress.append(current_balance)
        current_balance = portfolio_agent.getBalance()
        balance_difference = portfolio_agent.profit - balance_difference
        balance_difference_lst.append(balance_difference)
        balance_difference = portfolio_agent.profit

        if len(neuralNet.memory) > batch_size:  # and t != 0 and t % 4 == 0:
            neuralNet.batch_train(batch_size)
        i += 1
        if t == data_samples - 1:
            done = True
    print(portfolio_agent.profit)
    xaxis = np.array([i for i in range(data_samples)])
    # xaxis = np.array([portfolio_agent.stock_market[stock_names[0]].stock_data.index[i + start_date_ind]
    #                   for i in range(data_samples)])
    # xaxis = np.array(xaxis)
    y1axis = np.array(balance_progress)
    balance_graph("neuralNet", xaxis, y1axis, graph_index, interval)
    yaxis = np.array(balance_difference_lst)
    reward_graph("neuralNet", xaxis, yaxis, graph_index, interval)

    reward_pre_stock_graph("neuralNet", xaxis, stock_reward, graph_index, stock_names, interval)
    return y1axis, xaxis


def run_trader_linear(portfolio, file, initial_balance, stock_names, interval):
    tmp = portfolio.stock_market[portfolio.min_stick_len]
    data_samples = tmp.row_len - 1 - tmp.time_stamp
    start_date_ind = tmp.time_stamp
    balance_difference_lst = []
    current_balance = initial_balance
    stock_reward = {name: [] for name in stock_names}
    balance_progress = []
    balance_difference = 0
    states_buy = [[] for i in stock_names]
    states_sell = [[] for i in stock_names]
    for _ in tqdm(range(data_samples)):
        print(
            f'The date: {portfolio.stock_market[portfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}')
        file.write(
            f'The date: {portfolio.stock_market[portfolio.stock_name_list[0]].stock_data["DATE"].iloc[_]}' + '\n')
        actions, reward = portfolio.linear_reward()
        for ind, name in enumerate(stock_names):
            a = actions[ind]
            if a > 0:
                states_buy[ind].append(portfolio.stock_market[name].time_stamp )
            elif a < 0:
                states_sell[ind].append(portfolio.stock_market[name].time_stamp)
            stock_reward[name].append(reward[ind])
        portfolio.update_portfolio()
        balance_progress.append(current_balance)
        current_balance = portfolio.getBalance()
        balance_difference = portfolio.profit - balance_difference
        balance_difference_lst.append(balance_difference)
        balance_difference = portfolio.profit

    xaxis = np.array([i for i in range(data_samples)])
    # xaxis = np.array(
    #     [portfolio.stock_market[stock_names[0]].stock_data.index[i + start_date_ind] for i in range(data_samples)])
    yaxis = np.array(balance_difference_lst)
    reward_graph("Extrapolation", xaxis, yaxis, -1, interval)
    y1axis = np.array(balance_progress)
    balance_graph("Extrapolation", xaxis, y1axis, -1, interval)
    portfolio.getBalance()
    print(portfolio.profit)
    reward_pre_stock_graph("Extrapolation", xaxis, stock_reward, -1, stock_names, interval)
    # buy_sell_graph("Extrapolation", portfolio, stock_names, states_buy, states_sell, -1, interval[0])
    return y1axis


def main_def(kind_agent, new_net, skip_train):
    if not os.path.exists("images"):
        os.mkdir("images")
    # vars for PortFolio
    stock_names = ["AAPL", "GOOGL", "NDAQ", "NVDA"]
    delta_day = 32

    interval = "30m"
    date_list = make_date_list(delta_day)
    stock_indices = {name: i for name, i in enumerate(stock_names)}
    initial_investment = 10000

    episodes = 5
    # vars for NeuralNetwork
    state_size = 7
    action_space = 11
    batch_size: int = 8

    with open("result.txt", 'w') as f:
        if kind_agent == "1" or kind_agent == "3":

            if new_net != "1":
                neural_net = NeuralNetwork(episodes=episodes, state_size=state_size, action_space=action_space, model_to_load="startToWork.h5")
            else:
                neural_net = NeuralNetwork(episodes=episodes, state_size=state_size, action_space=action_space)
            # if skip_train == "1":
            #     episodes = 0
            # for episode in range(1, episodes + 1):
            #     portfolio = PortFolio(initial_investment, stock_names, interval[0], date_list[:haf_len_date],
            #                           stock_indices, f, action_space)
            #     print("Episode: {}/{}".format(episode, episodes))
            #     f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            #     f.write("Episode: {}/{}".format(episode, episodes) + '\n')
            #     f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            #     run_trader(neural_net, portfolio, batch_size, stock_names, f, initial_investment, episode, interval)
            #     if episode % 5 == 0:
            #         neural_net.model.save("startToWork{}.h5".format(episode))

            print("Test NeuralNetwork")
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Test NeuralNetwork" + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            portfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices,
                                  f, action_space)
            ay_net, ax = run_trader(neural_net, portfolio, batch_size, stock_names, f, initial_investment, -1, interval)
            neural_net.model.save("startToWork.h5")
        if kind_agent == "2" or kind_agent == "3":
            print("Test Linear")
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            f.write("Test Linear" + '\n')
            f.write("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$\n")
            portfolio = PortFolio(initial_investment, stock_names, interval, date_list, stock_indices, f,
                                  action_space)
            ay_extrapol = run_trader_linear(portfolio, f, initial_investment, stock_names, interval)
        if kind_agent == "3":
            balance_graph_together(ax, ay_net, ay_extrapol, interval)




if __name__ == "__main__":
    # agent_kind = sys.argv[1]
    agent_kind = input('Press 1 for run Neural Net agent and 2 for Extrapolation agent or 3 for run both agents\n')
    while agent_kind != "1" and agent_kind != "2" and agent_kind != "3":
        agent_kind = input(
            'Please press 1 for run Neural Net agent and 2 for Extrapolation agent or 3 for run both agents\n')

    new_net = None
    skip_train = None
    if agent_kind == "1" or agent_kind == "3":
        new_net = input("Press 1 for run on new neural net O.W press enter\n")
        if new_net != "1":
            skip_train = input("Press 1 for skip the training phase O.W press enter\n")
    main_def(agent_kind, new_net, skip_train)
