
import math
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

    def random_policy(self, T: int = 10000) -> List:

        total_reward = 0
        best_reward = 0
        for x in range(T):
            # get the arm from the random arm selection policy
            arm_index = self.random_arm_selection_policy()
            total_reward += self.get_uniform_distribution_reward(arm_index)
            best_reward += arm_index

        return [total_reward/T, (best_reward - total_reward)/T]

    def ETE_policy(self, T: int = 10000, m: List = [1, 10, 20, 30, 40, 50]) -> List:

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

        return average_reward_and_regret_pair

    def greedy_policy(self, T: int = 10000, e: List = [0.1, 0.2, 0.3, 0.4, 0.5]) -> List:

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

                else:         # exploit
                    arm_index = average_arm_reward.index(
                        max(average_arm_reward))

                arm_count[arm_index] += 1
                reward = self.get_uniform_distribution_reward(arm_index)
                arm_reward[arm_index] += reward
                average_arm_reward[arm_index] = arm_reward[arm_index] / \
                    arm_count[arm_index]
                total_reward += reward
                best_reward += arm_index + 1

            average_reward_and_regret_pair.append(
                [total_reward/T, (best_reward - total_reward)/T])

        return average_reward_and_regret_pair

    def adaptive_epsilon_greedy_policy(self, T: int = 10000) -> List:

        total_reward = 0
        best_reward = 0
        arm_count = [0 for i in range(self.N)]
        arm_reward = [0 for i in range(self.N)]
        average_arm_reward = [0 for i in range(self.N)]
        for t in range(1, T+1):
            if random.random() < (T*math.log(t)/t)**(1/3):              # explore
                arm_index = self.random_arm_selection_policy()
            else:                                             # exploit
                arm_index = average_arm_reward.index(max(average_arm_reward))

            arm_count[arm_index] += 1
            reward = self.get_uniform_distribution_reward(arm_index)
            arm_reward[arm_index] += reward
            average_arm_reward[arm_index] = arm_reward[arm_index] / \
                arm_count[arm_index]
            total_reward += reward
            best_reward += arm_index + 1

        return [total_reward/T, (best_reward - total_reward)/T]

    def upper_confidence_bound_policy(self, T: int = 10000) -> List:

        total_reward = 0
        arm_count = [0 for i in range(self.N)]
        arm_reward = [0 for i in range(self.N)]
        average_arm_reward = [0 for i in range(self.N)]
        upper_bound = [0 for i in range(self.N)]
        best_reward = 0
        for t in range(1, self.N+1):

            arm_index = t - 1
            reward = self.get_uniform_distribution_reward(arm_index)

            arm_count[arm_index] += 1
            arm_reward[arm_index] += reward
            average_arm_reward[arm_index] = arm_reward[arm_index] / \
                arm_count[arm_index]
            upper_bound[arm_index] = average_arm_reward[arm_index] + \
                math.sqrt(2*math.log(t)/arm_count[arm_index])
            total_reward += reward
            best_reward += arm_index + 1

        for t in range(self.N+1, T+1):

            arm_index = upper_bound.index(max(upper_bound))
            reward = self.get_uniform_distribution_reward(arm_index)

            arm_count[arm_index] += 1
            arm_reward[arm_index] += reward
            average_arm_reward[arm_index] = arm_reward[arm_index] / \
                arm_count[arm_index]
            upper_bound[arm_index] = average_arm_reward[arm_index] + \
                math.sqrt(2*math.log(t)/arm_count[arm_index])
            total_reward += reward
            best_reward += arm_index + 1

        return [total_reward/T, (best_reward - total_reward)/T]
