import gym
from gym import spaces
import numpy as np


class CustomGridEnv(gym.Env):
    def __init__(self, w, h, L, p, r):
        super(CustomGridEnv, self).__init__()
        self.w = w
        self.h = h
        self.L = {(x, y): reward for x, y, reward in L}
        self.p = p
        self.r = r
        self.action_space = spaces.Discrete(4)  # 4 actions: 0=left, 1=down, 2=right, 3=up
        self.observation_space = spaces.Discrete(w * h)
        self.state = None

    def reset(self):
        self.state = (0, 0)
        return self._state_to_index(self.state)

    def step(self, action):
        x, y = self.state
        if np.random.rand() < self.p:  # Take intended action with probability p
            if action == 0 and x > 0:  # left
                x -= 1
            elif action == 1 and y < self.h - 1:  # down
                y += 1
            elif action == 2 and x < self.w - 1:  # right
                x += 1
            elif action == 3 and y > 0:  # up
                y -= 1
        else:  # Take random action with probability 1-p
            action = np.random.choice([0, 1, 2, 3])
            if action == 0 and x > 0:  # left
                x -= 1
            elif action == 1 and y < self.h - 1:  # down
                y += 1
            elif action == 2 and x < self.w - 1:  # right
                x += 1
            elif action == 3 and y > 0:  # up
                y -= 1

        self.state = (x, y)
        reward = self.L.get(self.state, self.r)
        done = (self.state in self.L)
        return self._state_to_index(self.state), reward, done, {}

    def _state_to_index(self, state):
        x, y = state
        return y * self.w + x

    def _index_to_state(self, index):
        x = index % self.w
        y = index // self.w
        return (x, y)

    def render(self, mode='human'):
        grid = np.zeros((self.h, self.w), dtype=str)
        grid[:, :] = '.'
        for (x, y), reward in self.L.items():
            grid[y, x] = 'R' if reward != 0 else '.'
        x, y = self.state
        grid[y, x] = 'A'
        for row in grid:
            print(' '.join(row))
        print()


# Parameters
w = 4
h = 3
L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]
p = 0.8
r = -0.04

# Create custom environment
env = CustomGridEnv(w, h, L, p, r)

# Q-learning parameters
alpha = 0.1  # learning rate
gamma = 0.99  # discount factor
epsilon = 0.1  # exploration rate
n_episodes = 1000

# Initialize Q-table
Q = np.zeros((env.observation_space.n, env.action_space.n))


# Q-learning algorithm
def q_learning(env, Q, alpha, gamma, epsilon, n_episodes):
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        while not done:
            # Choose action using epsilon-greedy strategy
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(Q[state])  # Exploit

            next_state, reward, done, _ = env.step(action)
            # Update Q-value
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])
            state = next_state
    return Q


# Train the agent
Q = q_learning(env, Q, alpha, gamma, epsilon, n_episodes)

# Derive policy from Q-table
policy = np.argmax(Q, axis=1)

# Mapping from action to arrow
action_to_arrow = {0: '←', 1: '↓', 2: '→', 3: '↑'}


# Print the policy
def print_policy(policy, w, h):
    policy_grid = np.array([action_to_arrow[action] for action in policy]).reshape((h, w))
    for row in policy_grid:
        print(' '.join(row))


# Function to print the value table
def print_value_table(Q, w, h):
    value_table = np.max(Q, axis=1).reshape((h, w))
    for row in value_table:
        print(' '.join(f'{value:.2f}' for value in row))


print("Policy Derived from Q-learning:")
print_policy(policy, w, h)

print("Value Table Derived from Q-learning:")
print_value_table(Q, w, h)


# Run the policy
def run_policy(env, policy, episodes=10):
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = policy[state]
            state, reward, done, _ = env.step(action)
            # Print policy and value table after each step
            print("Policy Derived from Q-learning:")
            print_policy(policy, w, h)
            print("Value Table Derived from Q-learning:")
            print_value_table(Q, w, h)
        if done:
            env.render()
            print(f"Episode finished with reward: {reward}")


run_policy(env, policy)
