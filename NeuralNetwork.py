from collections import deque
import random
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import load_model


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


class NeuralNetwork:
    def __init__(self, episodes, signal_rate, state_size, action_space, model_name="AITraderBest",
                 gamma=0.95, epsilon=1.0, epsilon_final=0.01, epsilon_decay=0.995, load_model=None):
        # Rates definitions
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.signal_rate = signal_rate
        self.epsilon_final = epsilon_final

        # Reward preparation
        self.state_size = state_size
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.model_name = model_name
        self.loss = huber_loss
        self.custom_objects = {"huber_loss": huber_loss}  # important for loading the model from memory
        if load_model is None:
            self.model = self.model_builder()
        else:
            self.model = self.load(load_model)

    def model_builder(self):
        """
        creat a new neural network that will be the model
        :return: tf.keras.models.Sequential() that is the model
        """
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Dense(units=32, activation='relu', input_dim=self.state_size))
        model.add(tf.keras.layers.Dense(units=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=self.action_space, activation='linear'))
        model.compile(loss="mse", optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))
        return model

    def load(self, model_name):
        """
        Load existing model for the object
        :param model_name: the name of the model that the object use
        :return: tf.keras.models to use
        """
        return load_model(model_name, custom_objects=self.custom_objects)

    def action(self, state):
        """
        Take a action from given possible action
        :param state: the state that the model in, and by it need to take the action
        :return: the action that the model predict is the best
        """
        if random.random() <= self.epsilon:
            return random.randrange(self.action_space)
        actions = self.model.predict(state)
        return np.argmax(actions)

    def batch_train(self, batch_size):
        """
        Train the model on previous experiences
        :param batch_size: the size of the batch to train
        """
        batch = random.sample(self.memory, batch_size)
        X_train, y_train = [], []
        for state, action, reward, next_state, done in batch:
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.amax(self.model.predict(next_state))

            q_values = self.model.predict(state)
            action_num = action + int(self.action_space / 2)
            q_values[0][action_num] = target
            X_train.append(state[0])
            y_train.append(q_values[0])

        loss = self.model.fit(np.array(X_train), np.array(y_train), epochs=1, verbose=0).history["loss"][0]

        if self.epsilon > self.epsilon_final:
            self.epsilon *= self.epsilon_decay

        return loss
