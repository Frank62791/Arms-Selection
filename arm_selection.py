
import numpy as np

class ArmSelection():
    
    def __init__(self, N: int = 10):
        self.N = N
        self.arms = []
        self.create_arms()



    def create_arms(self):
        for i in range(self.N):
            self.arms.append(i + 1)


    def get_uniform_distribution_reward(self, arm: int) -> float:
        
        return np.random.uniform(0, arm)   # arm 1 will be uniformly distributed in [0, 1] and arm 2 may be uniformly distributed in [0, 2], etc.

    
    def get_average_reward(self,arm:int, N: int = 1000) -> float:
        total_reward = 0
        for i in range(N):
            total_reward += self.get_uniform_distribution_reward(arm)
        return total_reward / N
        