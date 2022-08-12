from Stock import Stock
import tensorflow as tf

class NeuralNetwork:
    def __init__(self, gamma, epsilon, epsilon_decay, episodes, signal_rate, stock_list):
        # Rates definitions
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.signal_rate = signal_rate

        # Stock Market Relevant

        # Neural Network Model
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=))
        self.model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=))

