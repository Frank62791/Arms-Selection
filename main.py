
from arm_selection import ArmSelection

T = 10000               #change  T = 1000 and T = 1500 and T = 2000
armSelection = ArmSelection()
print(armSelection.random_policy(T=T))                                                                 
print(armSelection.ETE_policy(T=T))
print(armSelection.greedy_policy(T=T))
print(armSelection.adaptive_epsilon_greedy_policy(T=T))
print(armSelection.upper_confidence_bound_policy(T=T))





