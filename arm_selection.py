
import math
import random
from typing import List
import numpy as np
import matplotlib.pyplot as plt


class ArmSelection():

    def __init__(self, N: int = 10, T: int = 10000):
        self.N = N
        self.arms = []
        self.create_arms()
        self.x_axia = np.arange(T)
        self.T = T
        self.fig, (self.reward_plot, self.regret_plot) = plt.subplots(1, 2)

    def create_arms(self):
        for i in range(self.N):
            self.arms.append(i)

    def get_uniform_distribution_reward(self, arm: int) -> float:

        # arm 1 will be uniformly distributed in [0, 1] and arm 2 will be uniformly distributed in [0, 2], etc.
        return np.random.uniform(0, arm+1)

    def get_average_reward(self, arm: int, N: int = 5000) -> float:
        total_reward = 0
        for i in range(N):
            total_reward += self.get_uniform_distribution_reward(arm)
        return total_reward / N

    def random_arm_selection_policy(self):
        return np.random.randint(0, self.N)

    def random_policy(self) -> List:

        total_reward = 0
        best_reward = 0
        reward_plot = []
        regret_plot = []
        for x in range(self.T):
            # get the arm from the random arm selection policy
            arm_index = self.random_arm_selection_policy()
            total_reward += self.get_uniform_distribution_reward(arm_index)
            best_reward += arm_index
            reward_plot.append(total_reward/(x+1))
            regret_plot.append((best_reward - total_reward)/(x+1))

        self.reward_plot.plot(self.x_axia, reward_plot,
                              label="random")
        self.regret_plot.plot(self.x_axia, regret_plot,
                              label="random")
        print("random policy average_reward: " + str(total_reward/self.T) + ", average_regret: " +
              str((best_reward - total_reward)/self.T))

    def ETE_policy(self, m: List = [1, 10, 20, 30, 40, 50]) -> List:

        for x in range(len(m)):

            total_reward = 0
            best_reward = 0
            arm_count = [0 for i in range(self.N)]
            arm_reward = [0 for i in range(self.N)]
            average_arm_reward = [0 for i in range(self.N)]
            reward_plot = []
            regret_plot = []
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
                reward_plot.append(average_arm_reward[arm_index])
                regret_plot.append((best_reward - total_reward)/(y+1))

            for y in range(m[x], self.T):            # exploit the remaining T - m[x] times
                arm_index = average_arm_reward.index(max(average_arm_reward))
                arm_count[arm_index] += 1
                reward = self.get_uniform_distribution_reward(arm_index)
                arm_reward[arm_index] += reward
                # calculate the empirical reward for each arm
                average_arm_reward[arm_index] = arm_reward[arm_index] / \
                    arm_count[arm_index]
                total_reward += reward
                best_reward += arm_index + 1
                reward_plot.append(average_arm_reward[arm_index])
                regret_plot.append((best_reward - total_reward)/(y+1))
            self.reward_plot.plot(self.x_axia, reward_plot,
                                  label="ETE m=" + str(m[x]) + "")
            self.regret_plot.plot(self.x_axia, regret_plot,
                                  label="ETE m=" + str(m[x]) + "")

            print("ETE m=" + str(m[x]) + " average_reward: " + str(total_reward/self.T) + ", average_regret: " +
                  str((best_reward - total_reward)/self.T))

    def greedy_policy(self, e: List = [0.1, 0.2, 0.3, 0.4, 0.5]) -> List:

        for epsilon in e:

            total_reward = 0
            best_reward = 0
            arm_count = [0 for i in range(self.N)]
            arm_reward = [0 for i in range(self.N)]
            average_arm_reward = [0 for i in range(self.N)]
            reward_plot = []
            regret_plot = []
            for y in range(self.T):

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

                reward_plot.append(average_arm_reward[arm_index])
                regret_plot.append((best_reward - total_reward)/(y+1))

            self.reward_plot.plot(self.x_axia, reward_plot,
                                  label="greedy e=" + str(epsilon) + "")
            self.regret_plot.plot(self.x_axia, regret_plot,
                                  label="greedy e=" + str(epsilon) + "")
            print("greedy e=" + str(epsilon) + " average_reward: " + str(total_reward/self.T) + ", average_regret: " +
                  str((best_reward - total_reward)/self.T))

    def adaptive_epsilon_greedy_policy(self) -> List:

        total_reward = 0
        best_reward = 0
        arm_count = [0 for i in range(self.N)]
        arm_reward = [0 for i in range(self.N)]
        average_arm_reward = [0 for i in range(self.N)]
        reward_plot = []
        regret_plot = []
        for t in range(1, self.T+1):
            if random.random() < (self.T*math.log(t)/t)**(1/3):              # explore
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
            reward_plot.append(average_arm_reward[arm_index])
            regret_plot.append((best_reward - total_reward)/t)
        self.reward_plot.plot(self.x_axia, reward_plot,
                              label="adaptive epsilon greedy")
        self.regret_plot.plot(self.x_axia, regret_plot,
                              label="adaptive epsilon greedy")
        print("adaptive epsilon greedy average_reward: " + str(total_reward/self.T) + ", average_regret: " +
              str((best_reward - total_reward)/self.T))

    def upper_confidence_bound_policy(self) -> List:

        total_reward = 0
        arm_count = [0 for i in range(self.N)]
        arm_reward = [0 for i in range(self.N)]
        average_arm_reward = [0 for i in range(self.N)]
        upper_bound = [0 for i in range(self.N)]
        best_reward = 0
        reward_plot = []
        regret_plot = []
        for t in range(1, self.N+1):

            arm_index = t - 1
            reward = self.get_uniform_distribution_reward(arm_index)

            arm_count[arm_index] += 1
            arm_reward[arm_index] += reward
            average_arm_reward[arm_index] = arm_reward[arm_index] / \
                arm_count[arm_index]
            reward_plot.append(average_arm_reward[arm_index])
            upper_bound[arm_index] = average_arm_reward[arm_index] + \
                math.sqrt(2*math.log(t)/arm_count[arm_index])
            total_reward += reward
            best_reward += arm_index + 1
            regret_plot.append((best_reward - total_reward)/t)

        for t in range(self.N+1, self.T+1):

            arm_index = upper_bound.index(max(upper_bound))
            reward = self.get_uniform_distribution_reward(arm_index)

            arm_count[arm_index] += 1
            arm_reward[arm_index] += reward
            average_arm_reward[arm_index] = arm_reward[arm_index] / \
                arm_count[arm_index]
            reward_plot.append(average_arm_reward[arm_index])
            upper_bound[arm_index] = average_arm_reward[arm_index] + \
                math.sqrt(2*math.log(t)/arm_count[arm_index])
            total_reward += reward
            best_reward += arm_index + 1
            regret_plot.append((best_reward - total_reward)/t)
        self.reward_plot.plot(self.x_axia, reward_plot,
                              label="upper confidence bound")
        self.regret_plot.plot(self.x_axia, regret_plot,
                              label="upper confidence bound")
        print("upper confidence bound average_reward: " + str(total_reward/self.T) + ", average_regret: " +
              str((best_reward - total_reward)/self.T))

    def save_figure(self, file_name: str = "undefined"):

        self.reward_plot.set_xlabel('Number of Time: T')
        self.reward_plot.set_ylabel("reward")
        self.reward_plot.set_title(
            "average cumulative reward T = " + str(self.T) + "")
        self.reward_plot.legend()

        self.regret_plot.set_xlabel('Number of Time: T')
        self.regret_plot.set_ylabel("regret")
        self.regret_plot.set_title("average regret T = " + str(self.T) + " ")
        self.regret_plot.legend()

        plt.savefig(file_name + ".png")
        plt.show()

      #  plt.show()
