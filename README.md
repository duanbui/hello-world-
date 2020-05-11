
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from itertools import count
%matplotlib inline

num_bandits = 10
mean_rewards = np.random.normal(size=num_bandits)

def get_reward(bandit,  mean_rewards, t=None):
    if bandit >=0 and bandit <=num_bandits:
        return np.random.normal(loc=mean_rewards[bandit], scale=1)
    else:
        raise ValueError('bandit >=0 and <{}'.format(num_bandits))


'''Sample rewards from k-armed test bed'''
reward_dist_sample_size = 100
possible_rewards = []
for a in range(num_bandits):
    possible_rewards += [(get_reward(a, mean_rewards),a) for _ 
                             in range(reward_dist_sample_size)]
sample_rewards = pd.DataFrame(possible_rewards,
                              columns = ['reward', 'bandit'])
                              
                              
'''Plotting reward distribution'''

plt.figure(figsize=(15,8))
ax = sns.violinplot(x=sample_rewards["bandit"]+1, y=sample_rewards["reward"])
