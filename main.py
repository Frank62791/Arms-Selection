
from arm_selection import ArmSelection


armSelection = ArmSelection()
print(armSelection.random_policy(T=10000))                                                                   #change  T = 1000 and T = 1500 and T = 2000
print(armSelection.ETE_policy(T=10000))
print(armSelection.greedy_policy(T=10000))
print(armSelection.adaptive_epsilon_greedy_policy(T=10000))
print(armSelection.upper_confidence_bound_policy(T=10000))





