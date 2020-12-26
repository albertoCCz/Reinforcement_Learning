# 10-armed Bandit Problem
## Introduction
In this problem we have an action space with 10 actions. The reward obtained when selecting an action is choosen 
from a normal distribution with mean equals the true value of the selected action and unit variance. Before that,
we selected the true value for the actions according to a normal distribution with mean zero and unit variance.

We evaluate three methods: Greedy, e-greedy with e = 0.01, and e-greedy with e = 0.1. All the methods formed their
action value estimates using the sample-average technique.

## Results
As expected from the theory. Taking e = 0.1 allows for a faster exploration of the action-value space. The average
reward grows rapidly in the first steps and gets stabilized at 1.38. Taking a lower value for e, like e = 0.01,
makes the algorithm to explore the space slowly, taking it more time to reach the same average reward than the
previous method. However, the average reward overcome the previously obtained value for the long run. 

![Average Reward - 10-armed Bandit](https://github.com/albertoCCz/Reinforcement_Learning/blob/main/10-armed%20Bandit/Average_Rewards.png)
