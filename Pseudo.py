import numpy as np
import scipy.special as ss

def X_Cube(size):
    X = []
    for i in range(size):
        x1 = -1 + 2*(i+1) / size
        x2 = np.random.uniform(0, 1) 
        x3 = np.random.uniform(0, 1)
        X.append([x1, x2, x3])
    X = np.array(X)
    return X

def X_Roll(size):
    X = []
    for i in range(size):
        theta = (i+1) / int(size/2) * np.pi
        r = theta / (2 * np.pi) * 1
        x1 = r * np.cos(theta) 
        x2 = r * np.sin(theta) 
        x3 = np.random.uniform(0, 1)
        X.append([x1, x2, x3])
    X = np.array(X)
    return X

def y_beta(size):
    y = []
    mean = []
    median = []
    variance = []
    skew = []    
    for i in range(size):
        alpha = 1 + (i+1)/size
        beta = 4 - 3*(i+1)/size
        y.append(np.random.beta(alpha,beta))
        mean.append(alpha / (alpha+beta))
        median.append(ss.betaincinv(alpha,beta,0.5))
        variance.append(alpha*beta/(((alpha+beta)**2)*(alpha+beta+1)))
        skew.append(2*(beta-alpha)*((alpha+beta+1)**(0.5))/((alpha+beta+2)*((alpha*beta)**0.5)))
    y = np.array(y)
    mean = np.array(mean)
    median = np.array(median)
    variance = np.array(variance)
    skew = np.array(skew)
    return y, mean, median, variance, skew