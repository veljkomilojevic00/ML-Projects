# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 10:21:34 2023

@author: veljko
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
FIGSIZE = (12, 6)
ALPHA = 0.2
plt.close('all')

file_name = "svmData.csv"
data = pd.read_csv(file_name).to_numpy()
y = data[:, -1]
X = data[:, 0:2]
class_values = np.array(np.unique(y), dtype=int)
class_colors = ['blue', 'red']

def train_test_split(X, y, train_per, seed=None):
    n = y.shape[-1]
    n_train = int(np.ceil(n*train_per))
    np.random.seed(seed)
    index = np.arange(stop=n, dtype=int)
    np.random.shuffle(index)
    
    X_train = X[index[0:n_train]]
    y_train = y[index[0:n_train]]

    X_test = X[index[n_train::]]
    y_test = y[index[n_train::]]

    return X_train, y_train, X_test, y_test


X_train, y_train, X_test, y_test = train_test_split(X, y, 0.8, seed=123)


class SVM:
    def __init__(self, fit_type="dual"):
        self.__fit_type = fit_type
        return 
    
    def __primal_cost_function(self, theta):
        w = theta[0:self.__n]
        b = theta[self.__n]
        xi = theta[self.__n+1::]

        return 0.5*w.dot(w) + self.__C*np.sum(xi)

    def __primal_make_constraints(self, theta):
        cons = np.zeros(shape=(2*self.__m, ))

        for i in range(self.__m):
            cons[i] = self.__y[i]*(self.__X[i, :].dot(theta[0:self.__n])
                                   +theta[self.__n]) - 1 + theta[self.__n+1+i]
            
        for i in range(self.__m):
            cons[i+self.__m] = theta[self.__n+1+i]
        return cons
    
    def __primal_fit(self, verbose):
        ineq_cons = {'type': 'ineq', 'fun' : self.__primal_make_constraints}
        theta0 = np.random.rand(self.__n+self.__m+1)
        
        options = {"disp": verbose}
        res = minimize(self.__primal_cost_function, theta0, method='SLSQP',
                       constraints=[ineq_cons], options=options)
        
        self.__w = res.x[0:self.__n]
        self.__b = res.x[self.__n]
        self.__xi = res.x[self.__n+1::]
        self.__support_vectors_ind, = np.where(self.__xi>0)
        
    def __primal_predict(self, X):
        return np.sign(self.__w.dot(X)+self.__b)
        
    def __get_primal_parameters(self):
        return self.__w, self.__b, self.__xi, self.__support_vectors_ind
        
    
    
    
    def __kernel(self, x, z):
        return np.exp(-(x-z).dot(x-z)/2/(self.__sigma)**2)
        
    def __dual_cost_function(self, alpha):
        return 0.5*alpha.T @ self.__Q @ alpha - self.__e.dot(alpha)
    
    def __dual_make_constraints(self, alpha):
        return self.__y.dot(alpha)
    
    def __dual_generate_bounds(self):
        bounds = [None]*self.__m
        
        for i in range(self.__m):
            bounds[i] = (0, self.__C)
        
        return bounds
    
    def __dual_fit(self, verbose):
        bounds = self.__dual_generate_bounds()
        eq_cons = {'type': 'eq', 'fun' : self.__dual_make_constraints}
        alpha0 = np.zeros(shape=(self.__m,))
        
        self.__Q = np.zeros(shape=(self.__m, self.__m))
        
        for i in range(self.__m):
            for j in range(self.__m):
                self.__Q[i, j] = self.__y[i]*self.__y[j]* \
                    self.__kernel(self.__X[i, ::], self.__X[j, ::])
                
        self.__e = np.ones(shape=(self.__m,))
        
        options = {"disp": verbose}
        res = minimize(self.__dual_cost_function, alpha0, method='SLSQP',
                       constraints=[eq_cons],
                       bounds=bounds, options=options)
        
        self.__alpha = res.x
        alpha_tol = 1e-3
        self.__alpha[self.__alpha<alpha_tol] = 0
        ind = np.argmax((self.__alpha<self.__C)*(self.__alpha>0))
        temp = 0
        for j in range(self.__m):
            temp += self.__y[j]*self.__alpha[j]*self.__kernel(self.__X[j, :],
                                                              self.__X[ind, :])
        self.__b = self.__y[ind] - temp
        
        self.__support_vectors_ind = np.where(self.__alpha>0)[0]
        self.__dual_coef = self.__y[self.__support_vectors_ind]*\
            (self.__alpha[self.__support_vectors_ind])
        self.__support_vectors = self.__X[self.__support_vectors_ind]
        
        
        
    def __dual_predict(self, x):
        n = self.__support_vectors_ind.shape[-1]
        K = np.zeros(self.__support_vectors_ind.shape[-1])
        for i in range(n):
            K[i] = self.__kernel(self.__support_vectors[i], x)
        return np.sign(np.sum(self.__y[self.__support_vectors_ind]*
                              self.__alpha[self.__support_vectors_ind]*K) + self.__b)
    
    def __get_dual_parameters(self):
        return self.__support_vectors_ind, self.__support_vectors,  self.__dual_coef, self.__alpha, self.__b
    


    def predict(self, X):
        if self.__fit_type == "primal":
            return self.__primal_predict(X)
            
        elif self.__fit_type == "dual":
            return self.__dual_predict(X)
            
        
        
    def fit(self, X, y, C, sigma=0.3, verbose=False):
        self.__C = C
        self.__X = X
        self.__y = y
        self.__m = X.shape[0]
        self.__n = X.shape[-1]

        if self.__fit_type == "primal":
            self.__primal_fit(verbose=False)
            
        elif self.__fit_type == "dual":
            self.__sigma = sigma
            self.__dual_fit(verbose=False)
        return
    
    def get_parameters(self):
        if self.__fit_type == "primal":
            return self.__get_primal_parameters()
            
        elif self.__fit_type == "dual":
            return self.__get_dual_parameters()
            
    def get_fit_type(self):
        return self.__fit_type


def K_fold_cross_validation(X, y, K, svm, C, sigma=None, seed=None, schuffle=False):
    n = y.shape[-1]
    n_val = n//K
    np.random.seed(seed)
    index = np.arange(stop=n, dtype=int)
    hinge_loss = np.zeros(shape=C.shape)
    for i in range(C.shape[-1]):
        if schuffle:    
            np.random.shuffle(index)
        for j in range(K):
            index_tf = np.full((n,), False, dtype=bool)
            index_tf[j*n_val:(j+1)*n_val] = np.full((n_val,), True, dtype=bool)
            X_val = X[index[index_tf]]
            y_val = y[index[index_tf]]
            X_train = X[index[~index_tf]]
            y_train = y[index[~index_tf]]
            
            if svm.get_fit_type() == "primal":
                svm.fit(X_train, y_train, C[i])
                w, b, xi, __ = svm.get_parameters()
                
                for k in range(y_val.shape[0]):
                    gamma_i = y_val[k]*(w.dot(X_val[k, :]) + b)
                    xi_i = max(0, 1 - gamma_i)
                    hinge_loss[i] += xi_i
                    
            elif svm.get_fit_type() == "dual":
                svm.fit(X_train, y_train, C[i])
                support_vectors_ind, __,  __, alpha, b = svm.get_parameters()
                kern = np.zeros(support_vectors_ind.shape[-1])
                
                for k in range(y_val.shape[0]):
                    for l in range(support_vectors_ind.shape[-1]):
                        x = X_train[support_vectors_ind[l], :]
                        z = X_val[k]
                        kern[l] = np.exp(-(x-z).dot(x-z)/2/(sigma)**2)
                    gamma_i = np.sum(y_train[support_vectors_ind]*
                                          alpha[support_vectors_ind]*kern) + b
                    xi_i = max(0, 1 - gamma_i)
                    hinge_loss[i] += xi_i
                
                
    return hinge_loss


svm = SVM("primal")

C = np.logspace(start=-3, stop=3, num=10)
loss = K_fold_cross_validation(X_train, y_train, 5, svm, C, 123)

plt.figure(figsize=FIGSIZE)
plt.subplot(121)
plt.loglog(C, loss)
plt.xlabel("C")
plt.ylabel("loss")
plt.title("K-fold cross validation log-log scale")
plt.grid(alpha=ALPHA)

C_opt = C[np.argmin(loss)]
print("Optimal hyperparameter C in first search: ", C_opt)

tol = C_opt/2
C = np.linspace(start=C_opt-tol, stop=C_opt+tol, num=20)
loss = K_fold_cross_validation(X_train, y_train, 5, svm, C, seed=123)

plt.subplot(122)
plt.plot(C, loss)
plt.xlabel("C")
plt.ylabel("loss")
plt.title("K-fold cross validation linear scale")
plt.grid(alpha=ALPHA)

C_opt = C[np.argmin(loss)]
print("Optimal hyperparameter C: ", C_opt)


svm.fit(X_train, y_train, C_opt)
w, b, xi, __ = svm.get_parameters()
xi_tol = 1e-3
xi[xi<xi_tol] = 0

x = np.linspace(0, 2, 100)
y = -w[0]/w[1]*x - b/w[1]


# define bounds of the domain
min1, max1 = X_train[:, 0].min()-1, X_train[:, 0].max()+1
min2, max2 = X_train[:, 1].min()-1, X_train[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 1e-2)
x2grid = np.arange(min2, max2, 1e-2)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))


yhat = np.zeros(shape=(grid.shape[0], ))
for i in range(yhat.shape[-1]):
    yhat[i]=svm.predict(grid[i, :])


zz = yhat.reshape(xx.shape)

plt.figure(figsize=FIGSIZE)
for class_value in range(2):
    plt.scatter(X_train[y_train==class_values[class_value], 0],
                X_train[y_train==class_values[class_value], 1],
                label="class: "+str(class_value),
                c=class_colors[class_value])
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='jet', alpha=ALPHA)

__, __, xi, support_vectors_ind = svm.get_parameters()
for i in support_vectors_ind:
        plt.annotate(str(xi[i])[0:5], (X_train[i, 0], X_train[i, 1]))
        
plt.scatter(X_train[support_vectors_ind, 0], X_train[support_vectors_ind, 1], c='yellow',
            marker='x', alpha=1, label="support vector")

plt.legend()
plt.title("Primal problem training set")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(alpha=ALPHA)


#%%
svm = SVM("dual")
sigma = 1.2
C = np.logspace(start=-3, stop=3, num=10)
loss = K_fold_cross_validation(X_train, y_train, 5, svm, C, sigma=sigma, seed=123)

plt.figure(figsize=FIGSIZE)
plt.subplot(121)
plt.loglog(C, loss)
plt.xlabel("C")
plt.ylabel("loss")
plt.title("K-fold cross validation log-log scale")
plt.grid(alpha=ALPHA)

C_opt = C[np.argmin(loss)]
print("Optimal hyperparameter C in first search: ", C_opt)

tol = C_opt/2
C = np.linspace(start=C_opt-tol, stop=C_opt+tol, num=20)
loss = K_fold_cross_validation(X_train, y_train, 5, svm, C, 123)

plt.subplot(122)
plt.plot(C, loss)
plt.xlabel("C")
plt.ylabel("loss")
plt.title("K-fold cross validation linear scale")
plt.grid(alpha=ALPHA)

C_opt = C[np.argmin(loss)]
print("Optimal hyperparameter C: ", C_opt)


svm.fit(X_train, y_train, C=C_opt, sigma=sigma)

# define bounds of the domain
min1, max1 = X_train[:, 0].min()-1, X_train[:, 0].max()+1
min2, max2 = X_train[:, 1].min()-1, X_train[:, 1].max()+1
# define the x and y scale
x1grid = np.arange(min1, max1, 1e-2)
x2grid = np.arange(min2, max2, 1e-2)
# create all of the lines and rows of the grid
xx, yy = np.meshgrid(x1grid, x2grid)
# flatten each grid to a vector
r1, r2 = xx.flatten(), yy.flatten()
r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
# horizontal stack vectors to create x1,x2 input for the model
grid = np.hstack((r1,r2))


yhat = np.zeros(shape=(grid.shape[0], ))
for i in range(yhat.shape[-1]):
    yhat[i]=svm.predict(grid[i, :])


zz = yhat.reshape(xx.shape)



plt.figure(figsize=FIGSIZE)
# plot the grid of x, y and z values as a surface
plt.contourf(xx, yy, zz, cmap='jet', alpha=ALPHA)

for class_value in range(2):
    plt.scatter(X_train[y_train==class_values[class_value], 0],
                X_train[y_train==class_values[class_value], 1],
                label="class: "+str(class_value),
                c=class_colors[class_value])

support_vectors_ind, __,  __, __, __ = svm.get_parameters()        
plt.scatter(X_train[support_vectors_ind, 0], X_train[support_vectors_ind, 1], c='yellow',
            marker='x', alpha=1, label="support vector")

plt.legend()
plt.title("Dual problem training set")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.grid(alpha=ALPHA)

plt.legend()
plt.show()