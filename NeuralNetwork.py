class NeuralNetwork:
    def __init__(self, gamma, epsilon, epsilon_decay, episodes, signal_rate, stock_list):
        # Rates definitions
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.signal_rate = signal_rate

        # Stock Market Relevant

        # Reward preparation
