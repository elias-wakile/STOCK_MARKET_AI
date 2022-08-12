from collections import deque
import tensorflow as tf
import random
import numpy as np
class NeuralNetwork:
    def __init__(self,  episodes, signal_rate, stock_list, state_size, action_space, model_name="AITraderBest",
                 gamma = 0.95, epsilon = 1.0, epsilon_final = 0.01, epsilon_decay = 0.995, ):
        # Rates definitions
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.signal_rate = signal_rate
        self.epsilon_final = epsilon_final

        # Stock Market Relevant

        # Reward preparation
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name
        self.first_iter = True
        self.model = self.model_builder()

    def model_builder(self):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=10, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=24, activation='relu'))
        model.add(tf.keras.layers.Dense(units=48, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss=self.loss, optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        if self.first_iter:
            self.first_iter = False
            return 1 # for case of trad in one stock and start with buy
        actions = self.model.predict(state)
        return np.argmax(actions)

    def batch_train(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])

            q_values = self.model.predict(state)
            q_values[0][action] = target
            X_train.append(state[0])
            y_train.append(q_values[0])


        loss = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0).history["loss"][0]

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        return loss
