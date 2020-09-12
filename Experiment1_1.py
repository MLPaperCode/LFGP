###############################################################################
# Experiment1_1 (figure1, figure2, figure3)
###############################################################################
size = 10000
size_test = 30
n_min = 1000 
epsilon = 1 
n_dim = None
n_neighbors = 50
omega_max = 100
map_X_list = [None, 'lle', 'isomap', 'umap']
map_y_list = ['mean', 'median', 'variance', 'skew']
X_dis_list = ['Cube', 'Roll']
y_dis = 'beta'
###############################################################################
from LFGP import LikelihoodFreeGaussianProcess
from Pseudo import X_Cube, X_Roll, y_beta
import matplotlib.pyplot as plt

for X_dis in X_dis_list:
    if X_dis == 'Cube' :
        X = X_Cube(size)
        X_test = X_Cube(size_test)
    elif X_dis == 'Roll' :
        X = X_Roll(size)
        X_test = X_Roll(size_test)
    y, mean, median, variance, skew = y_beta(size)
    y_test, mean_test, median_test, variance_test, skew_test = y_beta(size_test)
    for map_y in map_y_list:
        if map_y == 'mean' :
            baseline = mean_test
        elif map_y == 'median' :
            baseline = median_test
        elif map_y == 'variance' :
            baseline = variance_test
        elif map_y == 'skew' :
            baseline = skew_test
        for map_X in map_X_list:
            LFGP = LikelihoodFreeGaussianProcess(map_X=map_X, map_y=map_y)
            LFGP.fit(X, y, n_min=n_min, epsilon=epsilon, n_dim=n_dim, 
                     n_neighbors=n_neighbors, omega_max=omega_max, cluster=True)
            y_myu, y_sigma = LFGP.predict(X_test)
            LFGP.fit(X, y, n_min=n_min, epsilon=epsilon, n_dim=n_dim, 
                     n_neighbors=n_neighbors, omega_max=omega_max, cluster=False)
            y_myu_, y_sigma_ = LFGP.predict(X_test)
            fig = plt.figure()
            plt.grid(which = "major", axis = "x", color = "black", 
                     linestyle = "--", linewidth = 0.1)
            plt.grid(which = "major", axis = "y", color = "black", 
                     linestyle = "--", linewidth = 0.1)            
            plt.plot(X_test[:,0], y_myu[:,0], label='LFGP',
                    marker="o", color="#FF1FEC", markersize=5, linestyle='None')
            plt.plot(X_test[:,0], y_myu_[:,0], label='Baseline',
                    marker="x", color="#20B9FF", markersize=5, linestyle='None')
            plt.plot(X_test[:,0], baseline, label='True', 
                    color="black", linewidth=1, linestyle='-')
            plt.legend(frameon=True)
            plt.xlabel("$x_{i,1}^*$")
            plt.ylabel("Mean of $f(x_i^*)$")
            if map_X is None or map_X == 'lle' :                
                if map_y == 'mean':
                    plt.title("Mean", fontsize=30)
                elif map_y == 'median':
                    plt.title("Median", fontsize=30)
                elif map_y == 'variance':
                    plt.title("Variance", fontsize=30)
                elif map_y == 'skew':
                    plt.title("Skew", fontsize=30)
            plt.xlim(-1.5, 1.5)
            plt.xticks([-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5])
            if map_y == 'variance' :
                plt.ylim(0, 0.1)
            elif map_y == 'skew' :
                plt.ylim(-1.0, 1.5)                
            else:
                plt.ylim(0, 1.0)
            if map_X is None:
                plt.savefig('./' + map_y + '_' + X_dis + '_' + y_dis  + '.eps')
            else:
                plt.savefig('./' + map_X + '_' + map_y + '_' + X_dis + '_' + y_dis  + '.eps')               
            plt.show()