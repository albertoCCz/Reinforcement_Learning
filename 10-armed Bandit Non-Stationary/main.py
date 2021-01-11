import numpy as np
import matplotlib.pyplot as plt

# 10-Armed Bandit Problem - Non Stationary
# =================================================================

np.random.seed(14)

# Problem conditions
n_actions = 10
repetitions = 10000
interactions = 1500
e_probability = 0.1
step = 0.1

# Results array
avg_reward = np.zeros((2, interactions), dtype=float)

# Loop for average
for i in range(repetitions):
	# Choose randomly the initial true action values
	true_action_value = np.random.normal(loc=0.0, scale=1.0, size=n_actions)

	# Initialize problem
	estimate_action_value_st = np.zeros(n_actions, dtype=float)
	estimate_action_value_nst = np.zeros(n_actions, dtype=float)
	n_st = np.zeros(n_actions, dtype=int)
	n_nst = np.zeros(n_actions, dtype=int)

	# Learning loop
	reward_memory_st = []
	reward_memory_nst = []
	for j in range(interactions):
		# Determine the action to take
		if np.random.random() < e_probability:
			action = np.random.randint(low=0, high=10, dtype=int)
			action_st = action
			action_nst = action
		else:
			options_st = ((estimate_action_value_st - max(estimate_action_value_st)) == 0).nonzero()[0]
			options_nst = ((estimate_action_value_nst - max(estimate_action_value_nst)) == 0).nonzero()[0]
			# Non-random st action
			if len(options_st) > 1:
				action_st = options_st[np.random.randint(low=0, high=len(options_st), dtype=int)]
			else:
				action_st = options_st[0]

			# Non-random nst action
			if len(options_nst) > 1:
				action_nst = options_nst[np.random.randint(low=0, high=len(options_nst), dtype=int)]
			else:
				action_nst = options_nst[0]

		# Give a reward for that action and save it
		reward_st = np.random.normal(loc=true_action_value[action_st], scale=1.0)
		reward_nst = np.random.normal(loc=true_action_value[action_nst], scale=1.0)
		reward_memory_st.append(reward_st)
		reward_memory_nst.append(reward_nst)

		# Add one to step count for selected actions
		n_st[action_st] += 1
		n_nst[action_nst] += 1

		# Update the estimate action values
		estimate_action_value_st[action_st] += (reward_st - estimate_action_value_st[action_st]) / n_st[action_st]
		estimate_action_value_nst[action_nst] += (reward_nst - estimate_action_value_nst[action_nst]) * step

		# Update true action values
		true_action_value += np.random.normal(loc=0.0, scale=0.01, size=n_actions)

	# Save results
	avg_reward += np.asarray((reward_memory_st, reward_memory_nst)) / repetitions

	# Show progress
	print("Repetition progress: " + str(i / repetitions * 100) + str("%"))


# Plot average reward for each method over interactions
plt.plot(np.arange(interactions, dtype=int), avg_reward[0], '-g', linewidth=0.8, label="Sample-average")
plt.plot(np.arange(interactions, dtype=int), avg_reward[1], '-r', linewidth=0.8, label="Constant parameter")
plt.xlabel("Steps")
plt.ylabel("Average Reward")
plt.legend()
plt.show()
