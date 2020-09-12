import numpy as np
import random
import time
import umap
import scipy
from sklearn.cluster import KMeans
from sklearn.manifold import Isomap, LocallyLinearEmbedding as LLE
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel

class LikelihoodFreeGaussianProcess: 
    
    def __init__(self, map_X=None, map_y='mean'):
        self.map_X = map_X
        self.map_y = map_y
    
    def dim(self, X):
        X = X[:,None] if X.ndim == 1 else X
        return X
    
    def mapping(self, X, n_dim, n_neighbors):
        if self.scale == True:
            self.sc = StandardScaler()
            X_ = self.sc.fit_transform(X)
        else:
            X_ = X        
        n_dim = X_.shape[1] if n_dim is None else n_dim
        if self.map_X == 'pca':
            self.map = PCA(n_components=n_dim)
        elif self.map_X == 'lle':
            self.map = LLE(n_components=n_dim, n_neighbors=n_neighbors)
        elif self.map_X == 'isomap':
            self.map = Isomap(n_components=n_dim, n_neighbors=n_neighbors)
        elif self.map_X == 'umap':
            self.map = umap.UMAP(n_components=n_dim, n_neighbors=n_neighbors)
        time0 = time.time()  
        if self.map_X is not None:
            X_ = self.map.fit_transform(X_)
        time_map = time.time() - time0
        return X_, time_map

    def fitting(self, X_core, y_core):
        kernel = ConstantKernel()
        kernel *= RBF(length_scale=np.ones(X_core.shape[1]))
        kernel += WhiteKernel()
        gp = GPR(kernel=kernel).fit(X_core, y_core)
        length = gp.kernel_.get_params()['k1__k2__length_scale']
        return gp, length
    
    def clustering(self, X_, y, n_min):
        xm = XMeans(n_min=n_min).fit(X_)
        label = xm.labels_
        X_core = xm.cluster_centers_                
        y_core = self.stat(y, label)
        return X_core, y_core 
        
    def fit(self, X, y, n_min=100, epsilon=1, n_dim=None, n_neighbors=5, 
            omega_max=20, display=True, scale=True, cluster=True):
        self.scale = scale        
        X_, time_map = self.mapping(self.dim(X), n_dim, n_neighbors)
        y = self.dim(y)
        self.length = np.ones(X_.shape[1])
        time0 = time.time()
        X_core, y_core = self.clustering(X_ , y, n_min)
        gp, length = self.fitting(X_core, y_core)
        if cluster == True:
            for omega_ in range(omega_max):
                theta = gp.kernel_.theta
                theta[1:1+X_.shape[1]] = 0
                self.length *= length
                X_core, y_core = self.clustering(X_ / self.length, y, n_min)
                gp, length = self.fitting(X_core, y_core)                
                epsilon_ = gp.log_marginal_likelihood()
                epsilon_ -= gp.log_marginal_likelihood(theta=theta)
                if display ==True:
                    print("omega:", omega_ + 1, ", epsilon: {:.2f}".format(epsilon_))
                if epsilon_ <= epsilon:
                    break
            omega = omega_ + 1
            kernel = ConstantKernel(np.exp(theta[0]))
            kernel *= RBF(length_scale=np.ones(X_.shape[1]))
            kernel += WhiteKernel(np.exp(theta[X_.shape[1]+1]))
            self.gp = GPR(kernel=kernel, optimizer=None).fit(X_core, y_core)
        else:
            omega = 1
            self.gp, length = self.fitting(X_core, y_core)
        time_fit = time.time() - time0
        if display ==True:
            print("length:", self.length)
            print("Mapping time: {:.2f}sec".format(time_map))
            print("Fitting time: {:.2f}sec".format(time_fit))
        return omega, time_map, time_fit

    def predict(self, X):
        if self.map_X is None:
            if self.scale == True:            
                X_map = self.sc.transform(self.dim(X)) / self.length
            else:
                X_map = self.dim(X) / self.length                
        else:
            if self.scale == True:                        
                X_map = self.map.transform(self.sc.transform(self.dim(X))) / self.length
            else:
                X_map = self.map.transform(self.dim(X)) / self.length                
        y_myu, y_std = self.gp.predict(X_map, return_std=True)
        return self.dim(y_myu), self.dim(y_std)
        
    def stat(self, y, label):
        label_ = list(set(label))
        if self.map_y == 'mean':
            return np.array([np.mean(y[label==l],axis=0) for l in label_])
        elif self.map_y == 'median':
            return np.array([np.median(y[label==l],axis=0) for l in label_])
        elif self.map_y == 'variance':
            return np.array([np.var(y[label==l],axis=0,ddof=0) for l in label_])
        elif self.map_y == 'skew':
            return np.array([scipy.stats.skew(y[label==l],axis=0) for l in label_])
        else:
            return np.array([np.percentile(
                             y[label==l],self.map_y,axis=0) for l in label_])


# Modify a code from https://gist.github.com/yasaichi/254a060eff56a3b3b858
class XMeans:
    
    def __init__(self, n_min=10):
        self.n_min = n_min

    def fit(self, X):
        km = KMeans(n_clusters=1).fit(X)
        if len(X) < self.n_min * 2:
            self.labels_ = km.labels_
            self.cluster_centers_ = km.cluster_centers_
        else:
            self.c_ = []
            c = self.Cluster.build(X, km, np.arange(0, len(X)))
            self.split(c)
            self.labels_ = np.empty(len(X), dtype=np.intp)
            for i, c in enumerate(self.c_):
                self.labels_[c.index] = i
            self.cluster_centers_ = np.array([c.center for c in self.c_])
        return self

    def split(self, clusters):        
        for cluster in clusters: 
            if cluster.size < self.n_min * 2:
                self.c_.append(cluster)
                continue
            km = KMeans(n_clusters=2).fit(cluster.data)
            c1, c2 = self.Cluster.build(cluster.data, km, cluster.index)
            if c1.size < self.n_min or c2.size < self.n_min:
                size = cluster.size
                X_ = cluster.data
                rnd = random.sample(range(size), k=size)
                label = [1 if rnd[i] < size / 2 else 0 for i in range(size)]
                label = np.array(label)
                km.labels_ = label
                km.cluster_centers_[0] = np.mean(X_[label==0], axis=0)
                km.cluster_centers_[1] = np.mean(X_[label==1], axis=0)                
                c1, c2 = self.Cluster.build(cluster.data, km, cluster.index)
            self.split([c1, c2])

    class Cluster:
        @classmethod
        def build(cls, X, km, index):
            labels = range(0, km.n_clusters)
            return tuple(cls(X, index, km, label) for label in labels)
        
        def __init__(self, X, index, km, label):
            self.data = X[km.labels_==label]
            self.index = index[km.labels_==label]
            self.size = self.data.shape[0]
            self.center = km.cluster_centers_[label]