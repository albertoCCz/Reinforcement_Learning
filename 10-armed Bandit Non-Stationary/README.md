# 10-armed Bandit Problem with non-stationary optimal action values
## Introduction
The basics of this problem are the same that in the [stationary problem](https://github.com/albertoCCz/Reinforcement_Learning/tree/main/10-armed%20Bandit) we saw previously. The difference is that now the optimal action values change in time, with each interaction really. Thus, we are going to study a new method to estimate the action values and then we'll compare it with the sample-average method we used in the stationary case.

## Exponential recency-weighted average (constant-size parameter) method
In the stationary problem, we took

![Sample-average update rule](https://github.com/albertoCCz/Reinforcement_Learning/blob/main/10-armed%20Bandit%20Non-Stationary/Update_rule_st.png)

as the update rule, where _n_ is the number of times an specific action was selected, _Q_ is the estimated action value and _R_ is the reward received. For the non-stationary problem we modify slightly this expression changing the factor _1/n_ by _alpha_, which can takes values between 0 and 1:

![Constant-size parameter update rule](https://github.com/albertoCCz/Reinforcement_Learning/blob/main/10-armed%20Bandit%20Non-Stationary/Update_rule_nst.png)

Analyzing it, we can understand why the name "exponential recency-weighted": the weight given to each reward decreases as the number of intervening rewards increases. The fact that _alpha_ might be different from _1/n_ could cause convergence problems (for _alpha_=_1/n_ the convergence is guaranteed).

## Results
Let's compare both methods in a non-stationary problem. We use the parameters:
- n_actions = 10          # Size of action space
- repetitions = 10000     # Number of times the experiment is repeated to compute the average
- interactions = 1500     # Number of times the agent choose an action in each experiment
- e_probability = 0.1     # Probability that the agent choose a random action
- step = 0.1              # _alpha_

We find that, although the sample-average method explore rapidly the action space and accumulates a bigger reward in the "early game", the constant-size parameter method perform better at the end of the experiments, meaning that it adapts better to a changing enviroment.

![Average Reward - 10-armed Bandit Non-stationary](https://github.com/albertoCCz/Reinforcement_Learning/blob/main/10-armed%20Bandit%20Non-Stationary/Average_Reward.png)
