import numpy as np
import matplotlib.pyplot as plt

# 10-Bandit Problem
# =================================================================

np.random.seed(14)

# Loop for distinct e probabilities
repetitions = 1000
interactions = 1500
avg_reward = np.zeros((3, interactions), dtype=float)
e = [0, 0.1, 0.01]
for k in range(3):
	e_probability = e[k]
	# Loop for average
	for j in range(repetitions):
		# Choose randomly the true action values
		true_action_value = np.random.normal(loc=0.0, scale=1.0, size=10)

		# Initialize problem
		estimate_action_value = np.zeros(10, dtype=float)
		n = np.zeros(10, dtype=float)

		# Learning loop
		reward_memory = []
		i = 1
		while True:
			# Determine the action to take
			if np.random.random() < e_probability:
				action = np.random.randint(low=0, high=10, dtype=int)
			else:
				options = ((estimate_action_value - max(estimate_action_value)) == 0).nonzero()[0]
				if len(options) > 1:
					action = options[np.random.randint(low=0, high=len(options), dtype=int)]
				else:
					action = options[0]

			# Give a reward for that action
			reward = np.random.normal(loc=true_action_value[action], scale=1.0)
			reward_memory.append(reward)

			# Add one to step count for that action
			n[action] += 1

			# Update estimate action value function
			estimate_action_value[action] += 1/n[action] * (reward - estimate_action_value[action])

			# Break the loop when condition is met
			if i >= interactions:
				# Update average reward
				avg_reward[k] += np.asarray(reward_memory) / repetitions
				break
			else:
				i += 1


# # Plot reward over interactions
plt.plot(np.arange(interactions, dtype=int), avg_reward[0], '-c', linewidth=0.8, label="e = " + str(e[0]) + " (greedy)")
plt.plot(np.arange(interactions, dtype=int), avg_reward[1], '-g', linewidth=0.8, label="e = " + str(e[1]))
plt.plot(np.arange(interactions, dtype=int), avg_reward[2], '-r', linewidth=0.8, label="e = " + str(e[2]))
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
