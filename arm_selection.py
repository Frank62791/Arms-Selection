
import numpy as np

class ArmSelection():
    
    def __init__(self, N: int = 10):
        self.N = N
        self.arms = []
        self.create_arms()




    def create_arms(self):
        for i in range(self.N):
            self.arms.append(i + 1)

