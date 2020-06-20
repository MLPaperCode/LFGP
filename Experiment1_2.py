###############################################################################
# Experiment1_2 orrespoding to table3
###############################################################################
size_list = [100000, 200000, 300000, 400000, 800000, 1200000, , 1600000] 
n_min_list = [100, 500]
trials = 10
delta = 1 
epsilon = 1 
omega_max = 20
map_X = None 
map_y = 'mean'
X_dis = 'Cube'
y_dis = 'beta'
###############################################################################

from LFGP import LikelihoodFreeGaussianProcess
from Pseudo import X_Cube, y_beta
import numpy as np

for size in size_list:
    for n_min in n_min_list:
        time_list = []
        omega_list = []
        for i in range(trials):            
            X = X_Cube(size)
            y, mean1, median1, variance1, skew1, mean2, median2, \
                                variance2, skew2 = y_beta(size, display=False)
            LFGP = LikelihoodFreeGaussianProcess(map_X=map_X, map_y=map_y)
            omega, time_map, time_fit = LFGP.fit(X, y, n_min=n_min, 
                                            delta=delta, epsilon=epsilon, 
                                            omega_max=omega_max, display=False)
            if omega == omega_max:
                print("size:", size, "n_min:", n_min, 'Not converge')
                break
            else:        
                time_list += [time_fit]
                omega_list += [omega]
        print("size:", size, "n_min:", n_min, 
              "time:", np.mean(time_list), np.std(time_list, ddof=1), 
              "omega:", np.mean(omega_list), np.std(omega_list, ddof=1))
