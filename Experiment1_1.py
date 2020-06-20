###############################################################################
# Experiment1_1 correspoding to fig1, fig2
###############################################################################
size = 20000
size_test = 200
n_min = 100 
delta = 1 
epsilon = 1 
n_dim = None
n_neighbors = 30
omega_max = 20
map_X_list = [None, 'umap']
map_y_list = ['mean', 'median', 'variance', 'skew']
X_dis_list = ['Cube', 'Tube', 'Roll']
y_dis = 'beta'
###############################################################################

from LFGP import LikelihoodFreeGaussianProcess
from Pseudo import X_Cube, X_Tube, X_Roll, y_beta
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

convergence = []
for X_dis in X_dis_list:
    if X_dis == 'Cube' :
        X = X_Cube(size)
        X_test = X_Cube(size_test)
    elif X_dis == 'Tube' :
        X = X_Tube(size)
        X_test = X_Tube(size_test)
    elif X_dis == 'Roll' :
        X = X_Roll(size)
        X_test = X_Roll(size_test)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.plot(X[:int(size/2),0], X[:int(size/2),1], X[:int(size/2),2], 
            marker="o", color="#FF1FEC", markersize=5, linestyle='None')
    ax.plot(X[int(size/2):,0], X[int(size/2):,1], X[int(size/2):,2], 
            marker="x", color="#20B9FF", markersize=5, linestyle='None')
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel("$x_3$")
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(-2, 2)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.set_zticks([-2, -1, 0, 1, 2])
    plt.show()
    y, mean1, median1, variance1, skew1, mean2, median2, variance2, skew2 = y_beta(size)
    for map_y in map_y_list:
        if map_y == 'mean' :
            baseline1 = mean1
            baseline2 = mean2            
        elif map_y == 'median' :
            baseline1 = median1
            baseline2 = median2
        elif map_y == 'variance' :
            baseline1 = variance1
            baseline2 = variance2        
        elif map_y == 'skew' :
            baseline1 = skew1
            baseline2 = skew2        
        for map_X in map_X_list:
            LFGP = LikelihoodFreeGaussianProcess(map_X=map_X, map_y=map_y)
            omega, time_map, time_fit = LFGP.fit(X, y, n_min=n_min, delta=delta, 
                                                 epsilon=epsilon, n_dim=n_dim, 
                                                 n_neighbors=n_neighbors, omega_max=omega_max)
            y_myu, y_sigma = LFGP.predict(X_test)
            if omega == omega_max:
                if map_X is None:
                    convergence += [map_y + '_' + X_dis + '_' + y_dis]
                else:
                    convergence += [map_X + '_' + map_y + '_' + X_dis + '_' + y_dis]
            fig = plt.figure()
            plt.plot(X_test[:int(size_test/2),0], y_myu[:int(size_test/2),0], 
                    marker="o", color="#FF1FEC", markersize=5, linestyle='None')
            plt.plot(X_test[int(size_test/2):,0], y_myu[int(size_test/2):,0], 
                    marker="x", color="#20B9FF", markersize=5, linestyle='None')
            plt.xlabel("$x_{i,1}^*$")
            plt.ylabel("$f(x_i^*)$")
            if X_dis == 'Cube' and map_y == 'mean':
                plt.title("(a)", fontsize=30)
            elif X_dis == 'Cube' and map_y == 'median':
                plt.title("(b)", fontsize=30)
            elif X_dis == 'Cube' and map_y == 'variance':
                plt.title("(c)", fontsize=30)
            elif X_dis == 'Cube' and map_y == 'skew':
                plt.title("(d)", fontsize=30)
            plt.xlim(-2, 2)
            plt.xticks([-2, -1, 0, 1, 2])
            plt.hlines([baseline1], -2, 2, "#FF1FEC", linestyles='dashed')
            plt.hlines([baseline2], -2, 2, "#20B9FF", linestyles='dashed')
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
    print('Not converge', convergence)