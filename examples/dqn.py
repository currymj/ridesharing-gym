import gym
import ridesharing_gym
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, num_actions):
        self.state_size = state_size
        self.num_actions = num_actions
        self.memory = deque(maxlen=2000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.num_actions)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('ridesharing-v0')
    env.euclid = True
    raw_state = env.reset()
    def to_onehot(state):
        grid, requests = state
        requests_onehot = np.eye(len(grid))[requests]
        stacked = np.vstack( (grid, requests_onehot))
        return np.reshape(stacked.flatten(), (1, -1))
    state = to_onehot(raw_state).flatten()
    state_size = state.size
    print(state_size)
    num_actions = env.action_space.n
    agent = DQNAgent(state_size, num_actions)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        raw_state = env.reset()
        state = to_onehot(raw_state)

        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            print('action:', action, 'reward:', reward)
            print('total cars: {}'.format(np.sum(next_state[0])))
            reward = reward if not done else -10
            if action == 0:
                print('chose reject')
            next_state = to_onehot(next_state)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)



