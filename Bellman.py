import numpy as np


class BellmanSolver:
    def __init__(self, width, height, special_locations, default_reward, discount_factor, p=0.8):
        self.width = width
        self.height = height
        self.special_locations = {(x, y): reward for x, y, reward in special_locations}
        self.default_reward = default_reward
        self.discount_factor = discount_factor
        self.value_function = np.zeros((height, width))

        self.theta = 0.01
        self.p = p  # Probability of moving in the intended direction
        for i in special_locations:
            self.value_function[i[1], i[0]] = i[2]

    def is_within_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def get_reward(self, x, y):
        return self.special_locations.get((x, y), self.default_reward)

    def is_accessible(self, x, y):
        return (x, y) not in self.special_locations or self.special_locations[(x, y)] != 0

    def num_actions(self, x, y):
        counter = 0;
        ls = []
        print("num action " + str((x, y)))
        for action, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
            nx, ny = x + dx, y + dy
            if (self.is_within_bounds(nx, ny)):
                if not ((nx, ny) in self.special_locations and (self.special_locations[(nx, ny)] == 0)):
                    counter = counter + 1
                    ls.append((nx, ny))
        return (counter, ls)

    def bellman_update(self):
        new_value_function = np.copy(self.value_function)
        for y in range(self.height):
            for x in range(self.width):
                if not (x, y) in self.special_locations:
                    numa, num_ls = self.num_actions(x, y)
                    # print(x,y)
                    # print(self.value_function)
                    # print("numa " + str(numa))
                    # print("numls" + str(num_ls))

                    action_values = []
                    for action, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                        nx, ny = x + dx, y + dy
                        if self.is_within_bounds(nx, ny):
                            unintended_value = 0
                            intended_value = self.p * self.value_function[ny, nx]
                            for lx, ly in num_ls:
                                if (lx, ly) != (nx, ny):
                                    print("loo" + str((lx, ly)) + str((nx, ny)))
                                    print("yasoo")
                                    unintended_value = unintended_value + ((1 - self.p) / numa) * new_value_function[
                                        ly, lx]
                            action_value = intended_value + unintended_value
                        else:
                            action_value = self.p * self.value_function[y, x] + (1 - self.p) * new_value_function[y, x]
                        action_values.append(action_value)
                    print("size " + str(action_values))

                    max_action_value = max(action_values)
                    print("max " + str(action_value))
                    new_value_function[y, x] = self.get_reward(x, y) + self.discount_factor * max_action_value
                    print(new_value_function)
        return new_value_function

    def solve(self):
        while True:
            new_value_function = self.bellman_update()
            if np.max(np.abs(new_value_function - self.value_function)) < self.theta:
                break
            self.value_function = new_value_function

    def get_policy(self):
        policy = np.zeros((self.height, self.width), dtype=int)
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.special_locations:
                    continue
                action_values = []
                for action, (dx, dy) in enumerate([(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    nx, ny = x + dx, y + dy
                    if self.is_within_bounds(nx, ny):
                        if not ((nx, ny) in self.special_locations and (self.special_locations[(nx, ny)] == 0)):
                            action_value = self.value_function[ny, nx]  # Consider probability p
                            action_values.append((action_value, action))
                if action_values:
                    best_action = max(action_values, key=lambda x: x[0])[1]
                    policy[y, x] = best_action
        return policy

    def print_value_function(self):
        for row in self.value_function:
            print(" ".join([f"{v:.2f}" for v in row]))
        print()

    def print_policy(self):
        policy = self.get_policy()
        action_symbols = {0: '←', 1: '→', 2: '↑', 3: '↓'}
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) in self.special_locations:
                    if self.special_locations[(x, y)] == 0:
                        print('W', end=' ')
                    else:
                        print(self.special_locations[(x, y)], end=' ')

                else:
                    print(action_symbols[policy[y, x]], end=' ')
            print()
        print()


# Input parameters

w = 4
h = 3
L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]
p = 0.8
r = -0.04

counter = 0
for i in L:
    val = list(i)
    val[1] = h - val[1] - 1
    L[counter] = tuple(val)
    counter = counter + 1
solver = BellmanSolver(w, h, L, r, p)
solver.solve()
print("Analytical Solution (Value Function):")
solver.print_value_function()
print("Policy Derived from Value Function:")
solver.print_policy()
