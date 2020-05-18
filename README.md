
class NonSationaryEnv:
    
    def __init__(self,envtype = 'abrupt', num_bandits=10, max_time_steps=1000, num_change_points=None):
        self.env_type=envtype
        self.num_bandits = num_bandits
        self.T = max_time_steps
        self.change_point_indx = 0
        
        if num_change_points is None:
            self.num_change_points = int(np.log(self.T))
        else:
            self.num_change_points = num_change_points
        
        self.change_points = np.sort(np.random.randint(low=0, high=self.T-1,size=self.num_change_points))       
        self.mean_rewards = self._create_bandit_means_nonstationary()
        
    def _create_bandit_means_nonstationary(self):
        '''Choose independent random means for each abrupt change point'''
        return [np.random.normal(size=self.num_bandits) for _ in range(self.num_change_points+1)]
    
    def reset(self):
        """Resets the environment. Typically called before each new experiment"""
        self.change_point_indx = 0
        self.mean_rewards = self._create_bandit_means_nonstationary()
        
    def get_reward(self, bandit, t):        
        if bandit >=0 and bandit <=self.num_bandits:
            if self.change_point_indx< self.num_change_points\
                    and t > self.change_points[self.change_point_indx]:
                self.change_point_indx+=1            
            return np.random.normal(loc=self.mean_rewards[self.change_point_indx-1][bandit], scale=1)
        else:
            raise ValueError('argument bandit must be integer >=0 and <={}'.format(self.num_bandits))
            

non_stationary_env = NonSationaryEnv(envtype='abrupt')

num_experiments = 2000
reward_averages_by_eps = []

for eps in [0.1, 0.01, 0]:
    reward_histories = np.zeros(non_stationary_env.T)
    for exp_number in range(num_experiments):
        non_stationary_env.reset()
        if exp_number%500 == 0:
            print('Running Experiment Itrn#{} , epsilon={}'.format(exp_number,eps))
        history = run_eps_greedy(non_stationary_env, eps=eps, gamma=0.1)
        reward_histories = np.add(history,reward_histories )
    reward_averages_by_eps.append(reward_histories/num_experiments)
