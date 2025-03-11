import numpy as np
import matplotlib.pyplot as plt


class NonStationaryBandit:
    def __init__(self, k=10, std_dev=0.01):
        self.k = k  # Number of arms
        self.q_true = np.zeros(k)  # True action values
        self.std_dev = std_dev  # Standard deviation for nonstationary updates

    def step(self):
        self.q_true += np.random.normal(0, self.std_dev, self.k)  # Random walk

    def get_reward(self, action):
        return np.random.normal(self.q_true[action], 1)  # Reward with noise


def run_experiment(epsilon, alpha=None, steps=10000, runs=2000):
    k = 10
    avg_rewards = np.zeros(steps)
    optimal_action_counts = np.zeros(steps)

    for run in range(runs):
        bandit = NonStationaryBandit(k)
        Q = np.zeros(k)  # Initial action value estimates
        N = np.zeros(k)  # Action counts

        for t in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(k)  # Exploration
            else:
                action = np.argmax(Q)  # Exploitation

            reward = bandit.get_reward(action)
            bandit.step()  # Update environment

            optimal_action = np.argmax(bandit.q_true)  # Best action at this step
            if action == optimal_action:
                optimal_action_counts[t] += 1
            avg_rewards[t] += reward

            N[action] += 1
            if alpha is None:
                Q[action] += (1 / N[action]) * (reward - Q[action])  # Sample average
            else:
                Q[action] += alpha * (reward - Q[action])  # Constant step-size

    avg_rewards /= runs
    optimal_action_counts = (optimal_action_counts / runs) * 100
    return avg_rewards, optimal_action_counts


# Run experiments
epsilon = 0.1
steps = 10000
sample_avg_rewards, sample_avg_optimal = run_experiment(epsilon, alpha=None, steps=steps)
constant_step_rewards, constant_step_optimal = run_experiment(epsilon, alpha=0.1, steps=steps)

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(sample_avg_rewards, label='Sample Average')
plt.plot(constant_step_rewards, label='Constant Step-size (α=0.1)')
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(sample_avg_optimal, label='Sample Average')
plt.plot(constant_step_optimal, label='Constant Step-size (α=0.1)')
plt.xlabel("Steps")
plt.ylabel("% Optimal Action")
plt.legend()

plt.show()
