import numpy as np
import random
from get_all_test_cases import parse_tests

class ModelBasedRL:
    def __init__(self, width, height, special_locations, p, default_reward, discount_factor):
        self.width = width
        self.height = height
        self.special_locations = {(x, y): reward for x, y, reward in special_locations}
        self.p = p
        self.default_reward = default_reward
        self.discount_factor = discount_factor

        self.actions = ['up', 'down', 'left', 'right']
        self.action_effects = {
            'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)
        }

        self.value_function = np.zeros((height, width))
        for (x, y), reward in self.special_locations.items():
            self.value_function[y, x] = reward

        self.transition_counts = {(x, y): {a: np.zeros((height, width)) for a in self.actions}
                                  for x in range(width) for y in range(height)}
        self.reward_counts = np.zeros((height, width))

    def is_terminal(self, x, y):
        return (x, y) in self.special_locations

    def next_state(self, x, y, action):
        dx, dy = self.action_effects[action]
        nx, ny = x + dx, y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            return nx, ny
        return x, y  # Stay in place if would move out of bounds

    def simulate_experience(self, num_episodes):
        for _ in range(num_episodes):
            x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)
            while self.is_terminal(x, y):
                x, y = random.randint(0, self.width - 1), random.randint(0, self.height - 1)

            while not self.is_terminal(x, y):
                action = random.choice(self.actions)
                intended_nx, intended_ny = self.next_state(x, y, action)

                if random.random() < self.p:
                    nx, ny = intended_nx, intended_ny
                else:
                    # Random unintended action
                    nx, ny = self.next_state(x, y, random.choice(self.actions))

                reward = self.special_locations.get((nx, ny), self.default_reward)
                self.transition_counts[(x, y)][action][ny, nx] += 1
                self.reward_counts[ny, nx] = reward

                x, y = nx, ny

    def estimate_model(self):
        transition_probs = {(x, y): {a: np.zeros((self.height, self.width)) for a in self.actions}
                            for x in range(self.width) for y in range(self.height)}

        for (x, y), action_counts in self.transition_counts.items():
            for action, counts in action_counts.items():
                total = counts.sum()
                if total > 0:
                    transition_probs[(x, y)][action] = counts / total

        return transition_probs, self.reward_counts

    def value_iteration(self, max_iterations=1000, tolerance=0.01):
        for _ in range(max_iterations):
            new_value_function = np.copy(self.value_function)
            delta = 0

            for x in range(self.width):
                for y in range(self.height):
                    if self.is_terminal(x, y):
                        continue

                    v = self.value_function[y, x]
                    action_values = []

                    for action in self.actions:
                        action_value = sum(self.transition_probs[(x, y)][action][ny, nx] * self.value_function[ny, nx]
                                           for nx in range(self.width) for ny in range(self.height))
                        action_values.append(action_value)

                    new_value_function[y, x] = self.reward_counts[y, x] + self.discount_factor * max(action_values)
                    delta = max(delta, abs(v - new_value_function[y, x]))

            self.value_function = new_value_function
            if delta < tolerance:
                break

    def get_policy(self):
        policy = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if not self.is_terminal(x, y):
                    action_values = [sum(self.transition_probs[(x, y)][action][ny, nx] * self.value_function[ny, nx]
                                         for nx in range(self.width) for ny in range(self.height))
                                     for action in self.actions]
                    policy[y, x] = np.argmax(action_values)
        return policy

    def print_value_function(self):
        for row in self.value_function:
            print(" ".join([f"{v:.2f}" for v in row]))
        print()

def print_policy(policy, height, width):
    action_symbols = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    for y in range(height):
        for x in range(width):
            print(f"{action_symbols.get(policy[y, x], 'T'):^3}", end=' ')  # Center align the symbols
        print()
    print()

def run_model_based_solver(test_case):
    w, h, L, p, r = test_case['w'], test_case['h'], test_case['L'], test_case['p'], test_case['r']
    L = [(x, h - y - 1, reward) for x, y, reward in L]  # Adjust coordinates
    discount_factor = 0.9

    model = ModelBasedRL(w, h, L, p, r, discount_factor)
    model.simulate_experience(10000)
    model.transition_probs, _ = model.estimate_model()
    model.value_iteration()
    return model.value_function, model.get_policy()

if __name__ == "__main__":
    tests = parse_tests()
    results = []

    for i, test in enumerate(tests, start=1):
        value_function, policy = run_model_based_solver(test)
        print(f"Grid shape: {value_function.shape}")
        results.append((value_function, policy))

        print(f"Test {i} Value Function:")
        for row in value_function:
            print(" ".join([f"{v:.4f}" for v in row]))

        print(f"Test {i} Policy:")
        print_policy(policy, test['h'], test['w'])

        print("-"*40 + "\n")
