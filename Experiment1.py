###############################################################################
# Experiment1
# Correspoding to Figure 1
# Install UMAP for using LFGP. (pip install umap-learn)
###############################################################################
size = 10000
size_test = 200
map_X_list = [None, 'umap']
map_y_list = ['mean', 'median', 'variance']
X_dis_list = ['Cube', 'Tube']
y_dis = 'beta'
###############################################################################

from LFGP import LikelihoodFreeGaussianProcess
import numpy as np
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def X_Cube(size):
    X = []
    for i in range(size):
        x1 = np.random.uniform(0, 2) if i >= int(size/2) else np.random.uniform(-2, 0) 
        x2 = np.random.uniform(-2, 2) 
        x3 = np.random.uniform(-2, 2)
        X.append([x1, x2, x3])
    X = np.array(X)
    return X

def X_Tube(size):
    X = []
    for i in range(size):
        r = 2 if i >= int(size/2) else 1
        theta = i / int(size/2) * 2 * np.pi
        x1 = r * np.cos(theta) + np.random.normal(0,0.1)
        x2 = r * np.sin(theta) + np.random.normal(0,0.1)
        x3 = np.random.uniform(-2, 2)
        X.append([x1, x2, x3])
    X = np.array(X)
    return X

def y_beta(size):
    alpha1 = 1.0
    beta1 = 4.0
    alpha2 = 2.0
    beta2 = 1.0
    y = []
    for i in range(size):
        if i >= int(size/2):
            y.append(np.random.beta(alpha2,beta2))
        else:
            y.append(np.random.beta(alpha1,beta1))
    y = np.array(y)
    mean1 = alpha1 / (alpha1+beta1)
    median1 = scipy.special.betaincinv(alpha1,beta1,0.5)
    variance1 = alpha1*beta1 / (((alpha1+beta1)**2)*(alpha1+beta1+1))
    mean2 = alpha2 / (alpha2+beta2)
    median2 = scipy.special.betaincinv(alpha2,beta2,0.5)
    variance2 = alpha2*beta2 / (((alpha2+beta2)**2)*(alpha2+beta2+1))    
    print("Class1:", "mean", mean1, "median" ,median1, "variance", variance1)
    print("Class2:", "mean", mean2, "median" ,median2, "variance", variance2)
    return y, mean1, median1, variance1, mean2, median2, variance2

for X_dis in X_dis_list:
    if X_dis == 'Tube' :
        X = X_Tube(size)
        X_test = X_Tube(size_test)
    elif X_dis == 'Cube' :
        X = X_Cube(size)
        X_test = X_Cube(size_test)
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
    plt.savefig('./' + X_dis + '.eps')
    plt.show()
    y, mean1, median1, variance1, mean2, median2, variance2 = y_beta(size)
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
        for map_X in map_X_list:
            LFGP = LikelihoodFreeGaussianProcess(map_X=map_X, map_y=map_y)
            LFGP.fit(X, y)
            y_myu, y_sigma = LFGP.predict(X_test)
            fig = plt.figure()
            plt.plot(X_test[:int(size_test/2),0], y_myu[:int(size_test/2),0], 
                    marker="o", color="#FF1FEC", markersize=5, linestyle='None')
            plt.plot(X_test[int(size_test/2):,0], y_myu[int(size_test/2):,0], 
                    marker="x", color="#20B9FF", markersize=5, linestyle='None')
            plt.xlabel("$x_1$")
            plt.ylabel("$u$")
            plt.xlim(-2, 2)
            plt.xticks([-2, -1, 0, 1, 2])
            plt.hlines([baseline1], -2, 2, "gray", linestyles='dashed')
            plt.hlines([baseline2], -2, 2, "gray", linestyles='dashed')
            if map_y == 'variance' :
                plt.ylim(0, 0.1)
            else:
                plt.ylim(0, 1.0)
            if map_X is None:
                plt.savefig('./' + map_y + '_' + X_dis + '_' + y_dis  + '.eps')
            else:
                plt.savefig('./' + map_X + '_' + map_y + '_' + X_dis + '_' + y_dis  + '.eps')               
            plt.show()