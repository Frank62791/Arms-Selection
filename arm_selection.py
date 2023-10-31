
import random
from typing import List
import numpy as np


class ArmSelection():

    def __init__(self, N: int = 10):
        self.N = N
        self.arms = []
        self.create_arms()

    def create_arms(self):
        for i in range(self.N):
            self.arms.append(i)

    def get_uniform_distribution_reward(self, arm: int) -> float:

        # arm 1 will be uniformly distributed in [0, 1] and arm 2 may be uniformly distributed in [0, 2], etc.
        return np.random.uniform(0, arm+1)

    def get_average_reward(self, arm: int, N: int = 5000) -> float:
        total_reward = 0
        for i in range(N):
            total_reward += self.get_uniform_distribution_reward(arm)
        return total_reward / N

    def random_arm_selection_policy(self):
        return np.random.randint(0, self.N)

    def get_average_rewarrd_and_regret_by_random_policy(self, T: int = 10000):

        total_reward = 0
        best_reward = 0
        for x in range(T):
            # get the arm from the random arm selection policy
            arm_index = self.random_arm_selection_policy()
            total_reward += self.get_uniform_distribution_reward(arm_index)
            best_reward += arm_index

        return total_reward/T, (best_reward - total_reward)/T

    def get_average_reward_and_regret_by_ETE_policy(self, T: int = 10000, m: List = [1, 10, 20, 30, 40, 50]) -> List:

        average_reward_and_regret_pair = []
        for x in range(len(m)):

            total_reward = 0
            best_reward = 0
            arm_count = [0 for i in range(self.N)]
            arm_reward = [0 for i in range(self.N)]
            average_arm_reward = [0 for i in range(self.N)]
            for y in range(0, m[x]):              # explore the first m[x] times
                arm_index = self.random_arm_selection_policy()
                arm_count[arm_index] += 1
                reward = self.get_uniform_distribution_reward(arm_index)
                arm_reward[arm_index] += reward
                # calculate the empirical reward for each arm
                average_arm_reward[arm_index] = arm_reward[arm_index] / \
                    arm_count[arm_index]
                total_reward += reward
                best_reward += arm_index + 1

            for y in range(m[x], T):            # exploit the remaining T - m[x] times
                arm_index = average_arm_reward.index(max(average_arm_reward))
                arm_count[arm_index] += 1
                reward = self.get_uniform_distribution_reward(arm_index)
                arm_reward[arm_index] += reward
                # calculate the empirical reward for each arm
                average_arm_reward[arm_index] = arm_reward[arm_index] / \
                    arm_count[arm_index]
                total_reward += reward
                best_reward += arm_index + 1 

            average_reward_and_regret_pair.append(
                [total_reward/T, (best_reward - total_reward)/T])

        return  average_reward_and_regret_pair
    
    
    def get_average_reward_and_regret_by_greedy_policy(self, T: int = 10000, e: List = [ 0.1, 0.2, 0.3, 0.4, 0.5 ]) -> List:
        
        average_reward_and_regret_pair = []
        for epsilon in e:

            total_reward = 0
            best_reward = 0
            arm_count = [0 for i in range(self.N)]
            arm_reward = [0 for i in range(self.N)]
            average_arm_reward = [0 for i in range(self.N)]
            for y in range(T):      
                
                if random.random() < epsilon:              # explore       
                    arm_index = self.random_arm_selection_policy()
                    arm_count[arm_index] += 1
                    reward = self.get_uniform_distribution_reward(arm_index)
                    arm_reward[arm_index] += reward
                    # calculate the empirical reward for each arm
                    average_arm_reward[arm_index] = arm_reward[arm_index] / \
                        arm_count[arm_index]
                    total_reward += reward
                    best_reward += arm_index + 1 

                else:         # exploit

                    arm_index = average_arm_reward.index(max(average_arm_reward))
                    arm_count[arm_index] += 1
                    reward = self.get_uniform_distribution_reward(arm_index)
                    arm_reward[arm_index] += reward
                    # calculate the empirical reward for each arm
                    average_arm_reward[arm_index] = arm_reward[arm_index] / \
                        arm_count[arm_index]
                    total_reward += reward
                    best_reward += arm_index + 1

            average_reward_and_regret_pair.append(
                [total_reward/T, (best_reward - total_reward)/T])
        
        return average_reward_and_regret_pair

