import numpy as np
import random

# Define grid world dimensions and parameters
w = 4
h = 3
L = [(1, 1, 0), (3, 2, 1), (3, 1, -1)]  # terminal states with rewards
p = 0.8  # probability of intended action
r = -0.04  # reward for non-terminal states
gamma = 0.9  # discount factor
tol = 0.01  # tolerance for stopping criteria

# Actions and their effects
actions = ['up', 'down', 'left', 'right']
action_vectors = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1)
}

# Initialize value function
V = np.zeros((h, w))

# Set terminal states
terminals = {(x-1, y-1): reward for x, y, reward in L}
for (x, y), reward in terminals.items():
    V[x, y] = reward

# Initialize transition and reward models
transition_counts = {(x, y): {a: np.zeros((h, w)) for a in actions} for x in range(h) for y in range(w)}
reward_counts = np.zeros((h, w))

# Function to get next state
def next_state(x, y, action):
    dx, dy = action_vectors[action]
    nx, ny = x + dx, y + dy
    if 0 <= nx < h and 0 <= ny < w:
        return nx, ny
    return x, y

# Simulate experience
def simulate_experience(num_episodes):
    for episode in range(num_episodes):
        x, y = random.randint(0, h-1), random.randint(0, w-1)
        while (x, y) in terminals:
            x, y = random.randint(0, h-1), random.randint(0, w-1)
        while (x, y) not in terminals:
            action = random.choice(actions)
            nx, ny = next_state(x, y, action)
            if random.random() < p:
                next_x, next_y = nx, ny
            else:
                next_x, next_y = next_state(x, y, random.choice(actions))
            reward = terminals.get((next_x, next_y), r)
            transition_counts[(x, y)][action][next_x, next_y] += 1
            reward_counts[next_x, next_y] = reward
            x, y = next_x, next_y

# Estimate transition probabilities and rewards from counts
def estimate_model():
    transition_prob = {(x, y): {a: np.zeros((h, w)) for a in actions} for x in range(h) for y in range(w)}
    for (x, y), action_counts in transition_counts.items():
        for action, counts in action_counts.items():
            total = counts.sum()
            if total > 0:
                transition_prob[(x, y)][action] = counts / total
    return transition_prob, reward_counts

# Value Iteration
def value_iteration(V, gamma, estimated_transitions, estimated_rewards, terminals, actions, max_iterations=1000, tol=0.01):
    for iteration in range(max_iterations):
        new_V = np.copy(V)
        delta = 0
        for x in range(h):
            for y in range(w):
                if (x, y) in terminals:
                    continue
                v = V[x, y]
                action_values = []
                for action in actions:
                    action_value = sum(estimated_transitions[(x, y)][action][nx, ny] * V[nx, ny] for nx in range(h) for ny in range(w))
                    action_values.append(action_value)
                new_V[x, y] = estimated_rewards[x, y] + gamma * max(action_values)
                delta = max(delta, abs(v - new_V[x, y]))
        V = new_V
        print(f"Iteration {iteration + 1}:")
        print(V)
        if delta < tol:
            break
    return V

# Simulate experience to gather data
simulate_experience(10000)

# Estimate the model from the gathered data
estimated_transitions, estimated_rewards = estimate_model()

# Run value iteration using the estimated model
final_values = value_iteration(V, gamma, estimated_transitions, estimated_rewards, terminals, actions, tol=tol)
print("Final Value Function:")
print(final_values)
