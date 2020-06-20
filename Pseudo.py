import numpy as np
import scipy.special as ss

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

def X_Roll(size):
    X = []
    for i in range(size):
        theta = i / int(size/2) * np.pi
        r = theta / (2 * np.pi) * 2
        x1 = r * np.cos(theta) + np.random.normal(0,0.1)
        x2 = r * np.sin(theta) + np.random.normal(0,0.1)
        x3 = np.random.uniform(-2, 2)
        X.append([x1, x2, x3])
    X = np.array(X)
    return X

def y_beta(size, display=True):
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
    median1 = ss.betaincinv(alpha1,beta1,0.5)
    variance1 = alpha1*beta1 / (((alpha1+beta1)**2)*(alpha1+beta1+1))
    skew1 = 2*(beta1-alpha1)*((alpha1+beta1+1)**(0.5))/((alpha1+beta1+2)*((alpha1*beta1)**0.5))
    mean2 = alpha2 / (alpha2+beta2)
    median2 = ss.betaincinv(alpha2,beta2,0.5)
    variance2 = alpha2*beta2 / (((alpha2+beta2)**2)*(alpha2+beta2+1))    
    skew2 = 2*(beta2-alpha2)*((alpha2+beta2+1)**(0.5))/((alpha2+beta2+2)*((alpha2*beta2)**0.5))
    if display == True:
        print("Class1:", "mean", mean1, "median" ,median1, "variance", variance1, 'skew', skew1)
        print("Class2:", "mean", mean2, "median" ,median2, "variance", variance2, 'skew', skew2)
    return y, mean1, median1, variance1, skew1, mean2, median2, variance2, skew2