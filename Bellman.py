import numpy as np
from get_all_test_cases import parse_tests


class BellmanSolver:
    def __init__(self, width, height, special_locations, default_reward, discount_factor, p=0.8):
        self.width = width
        self.height = height
        self.special_locations = {(x, height - 1 - y): reward for x, y, reward in
                                  special_locations}  # Invert y-coordinate
        self.default_reward = default_reward
        self.discount_factor = discount_factor
        self.value_function = np.zeros((height, width))
        self.p = p
        self.theta = 0.01

        for (x, y), reward in self.special_locations.items():
            self.value_function[y, x] = reward

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_reward(self, x, y):
        return self.special_locations.get((x, y), self.default_reward)

    def is_accessible(self, x, y):
        return (x, y) not in self.special_locations or self.special_locations[(x, y)] != 0

    def get_valid_actions(self, x, y):
        valid_actions = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = x + dx, y + dy
            if self.is_within_bounds(nx, ny) and self.is_accessible(nx, ny):
                valid_actions.append((nx, ny))
        return valid_actions

    def bellman_update(self):
        new_value_function = np.copy(self.value_function)
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.special_locations:
                    valid_actions = self.get_valid_actions(x, y)
                    action_values = []

                    for nx, ny in valid_actions:
                        intended_value = self.p * self.value_function[ny, nx]
                        unintended_value = sum((1 - self.p) / len(valid_actions) * self.value_function[ly, lx]
                                               for lx, ly in valid_actions if (lx, ly) != (nx, ny))
                        action_values.append(intended_value + unintended_value)

                    if not action_values:  # If no valid actions, stay in place
                        action_values = [self.value_function[y, x]]

                    max_action_value = max(action_values)
                    new_value_function[y, x] = self.get_reward(x, y) + self.discount_factor * max_action_value

        return new_value_function

    def get_policy(self):
        policy = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in self.special_locations:
                    valid_actions = self.get_valid_actions(x, y)
                    if valid_actions:
                        action_values = [self.value_function[ny, nx] for nx, ny in valid_actions]
                        best_action = valid_actions[np.argmax(action_values)]
                        dx, dy = best_action[0] - x, best_action[1] - y
                        policy[y, x] = {(-1, 0): 2, (1, 0): 3, (0, -1): 0, (0, 1): 1}[(dx, dy)]
        return policy

    def solve(self):
        iteration = 0
        while True:
            iteration += 1
            new_value_function = self.bellman_update()
            if np.max(np.abs(new_value_function - self.value_function)) < self.theta:
                break
            self.value_function = new_value_function
        print(f"Converged after {iteration} iterations")


def print_policy(policy, height, width):
    action_symbols = {0: '←', 1: '→', 2: '↑', 3: '↓'}
    for y in range(height):
        for x in range(width):
            print(f"{action_symbols.get(policy[y, x], 'S'):^3}", end=' ')  # Center align the symbols
        print()
    print()


def run_bellman_solver(test_case):
    w, h, L, p, r = test_case['w'], test_case['h'], test_case['L'], test_case['p'], test_case['r']
    solver = BellmanSolver(w, h, L, r, 0.9, p)  # Use 0.9 as the discount factor
    solver.solve()
    policy = solver.get_policy()
    return solver.value_function, policy  # Flip the grid vertically to match the expected orientation


if __name__ == "__main__":
    tests = parse_tests()
    for i, test in enumerate(tests, 1):
        print(f"Test {i}:")
        value_function, policy = run_bellman_solver(test)
        print(f"Grid shape: {value_function.shape}")

        print("Value function:")
        for row in value_function:
            print(" ".join([f"{v:.4f}" for v in row]))

        print("\nPolicy:")
        print_policy(policy, test['h'], test['w'])

        print("\n" + "-" * 40)
