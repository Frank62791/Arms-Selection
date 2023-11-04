
from arm_selection import ArmSelection

T = 10000            #change  T = 1000 and T = 1500 and T = 2000
armSelection = ArmSelection(T=T)
armSelection.random_policy()
armSelection.ETE_policy()
armSelection.greedy_policy()                                                 
armSelection.adaptive_epsilon_greedy_policy()
armSelection.upper_confidence_bound_policy()
armSelection.save_figure()   







