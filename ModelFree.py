import numpy as np
import random
from get_all_test_cases import parse_tests

class GridWorld:
    def __init__(self, width, height, special_locations, p, default_reward):
        self.width = width
        self.height = height
        self.special_locations = {(x, y): reward for x, y, reward in special_locations}
        self.p = p
        self.default_reward = default_reward
        self.state = None
        self.actions = [0, 1, 2, 3]  # 0: up, 1: down, 2: left, 3: right

    def reset(self):
        self.state = (0, 0)
        return self._state_to_index(self.state)

    def step(self, action):
        x, y = self.state
        if random.random() < self.p:
            dx, dy = [(0, -1), (1, 0), (0, 1), (-1, 0)][action]
        else:
            dx, dy = random.choice([(0, -1), (1, 0), (0, 1), (-1, 0)])

        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            self.state = (nx, ny)

        reward = self.special_locations.get(self.state, self.default_reward)
        done = self.state in self.special_locations
        return self._state_to_index(self.state), reward, done

    def _state_to_index(self, state):
        x, y = state
        return y * self.width + x

    def _index_to_state(self, index):
        y, x = divmod(index, self.width)
        return (x, y)


class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((env.width * env.height, len(env.actions)))

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.actions)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, n_episodes):
        for episode in range(n_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                # Q-learning update
                best_next_action = np.argmax(self.q_table[next_state])
                td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
                td_error = td_target - self.q_table[state][action]
                self.q_table[state][action] += self.lr * td_error

                state = next_state

    def get_policy(self):
        policy = np.zeros((self.env.height, self.env.width), dtype=int)
        for y in range(self.env.height):
            for x in range(self.env.width):
                state = self.env._state_to_index((x, y))
                policy[y, x] = np.argmax(self.q_table[state])
        return policy

    def print_q_values(self):
        for y in range(self.env.height):
            for x in range(self.env.width):
                state = self.env._state_to_index((x, y))
                print(f"State ({x}, {y}):")
                for action, q_value in enumerate(self.q_table[state]):
                    print(f"  Action {action}: {q_value:.2f}")
                print()


def print_policy(policy, height, width, special_locations):
    action_symbols = {0: '↑', 1: '→', 2: '←', 3: '↓'}
    for y in range(height):
        for x in range(width):
            if (x, y) in special_locations:
                reward = special_locations[(x, y)]
                symbol = 'W' if reward == 0 else str(reward)
            else:
                symbol = action_symbols.get(policy[y, x], 'S')
            print(f"{symbol:^3}", end=' ')
        print()
    print()


def model_free_solver(test_case):
    w, h, L, p, r = test_case['w'], test_case['h'], test_case['L'], test_case['p'], test_case['r']
    L = [(x, h - y - 1, reward) for x, y, reward in L]  # Adjust coordinates

    env = GridWorld(w, h, L, p, r)
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    agent.learn(n_episodes=10000)

    value_function = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            state = env._state_to_index((x, y))
            if (x, y) in env.special_locations:
                value_function[y, x] = env.special_locations[(x, y)]
            else:
                value_function[y, x] = np.max(agent.q_table[state])

    policy = agent.get_policy()
    return value_function, policy, env.special_locations
def run_model_free_solver(test_case):
    w, h, L, p, r = test_case['w'], test_case['h'], test_case['L'], test_case['p'], test_case['r']
    L = [(x, h - y - 1, reward) for x, y, reward in L]  # Adjust coordinates

    env = GridWorld(w, h, L, p, r)
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1)
    agent.learn(n_episodes=10000)

    value_function = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            state = env._state_to_index((x, y))
            value_function[y, x] = np.max(agent.q_table[state])

    policy = agent.get_policy()
    return value_function, policy


if __name__ == "__main__":
    tests = parse_tests()
    results = []

    for i, test in enumerate(tests, 1):
        print(f"Test {i}:")
        print(f"Grid size: {test['w']}x{test['h']}")

        value_function, policy, special_locations = model_free_solver(test)
        results.append((value_function, policy))

        print("\nValue function:")
        for y, row in enumerate(value_function):
            for x, value in enumerate(row):
                if (x, y) in special_locations:
                    reward = special_locations[(x, y)]
                    print(f"{reward:7.4f}", end=" ")
                else:
                    print(f"{value:7.4f}", end=" ")
            print()

        print("\nPolicy:")
        print_policy(policy, test['h'], test['w'], special_locations)

        print("-" * 40)
