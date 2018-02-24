#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: logistic.py
#
'''
        ** Logistic Regression **

This is a logistic regression classifier example.

Sigmoid function is applied. 

Cross entropy has been choosen as Cost Function. 

The data are rescaled before training.

The gradient descent method is applied to train the weight theta.

'''

from numpy import loadtxt, where
import numpy as np
import matplotlib.pyplot as plt


def loadData():
    # load the dataset
    data = loadtxt('./data1.txt', delimiter=',')
    X = np.ones([100, 3])
    y = np.zeros(100)
    X[:, 1:3] = data[:, 0:2]
    y = data[:, 2]
    # rescale the raw data to a reasonable range.
    X[:,1] = (X[:,1] - sum(X[:,1])/100.) / (max(X[:,1]) - min(X[:,1]))
    X[:,2] = (X[:,2] - sum(X[:,2])/100.) / (max(X[:,2]) - min(X[:,2]))
    # plot the scaled data set
    pos = where(y == 1)
    neg = where(y == 0)
    plt.figure()
    plt.plot(X[pos, 1], X[pos, 2], marker='o', c='b')
    plt.plot(X[neg, 1], X[neg, 2], marker='o', c='r')
    plt.xlabel('feature1 (X1)')
    plt.ylabel('feature2 (X2)')
    plt.show()
    return X, y


def sigmoid(X):  
    # Compute sigmoid function
    gz =1.0 / (1.0+ np.exp(-1.0 * X))
    return gz  


def computeCost(theta,X,y):
    # compute cost function
    m = X.shape[0] #number of training examples
    J =(-1./m)*(np.transpose(y).dot(np.log(sigmoid(X.dot(theta))))
                + np.transpose(1-y).dot(np.log(1-sigmoid(X.dot(theta)))))
    return J


def computeGrad(theta, X, y):
    m = X.shape[0]
    grad = np.transpose((1. / m) * np.transpose(sigmoid(X.dot(theta)) - y).dot(X))
    return grad


def train(theta, X, y, alpha):
    print 'Start training ...'
    # apply the gradient descent optimization
    for i in range(0,1000):
        Jtmp = computeCost(theta, X, y)
        print 'cost function J(',i,') =', Jtmp
        grad = computeGrad(theta, X, y)
        theta = theta - alpha * grad
    print 'The training is finished.'
    print 'theta =', theta
    return theta


def predict(theta, X, y):
    # Predict label using learned logistic regression parameters
    m, n = X.shape  
    p = np.zeros(shape=(m,1))
    h = sigmoid(X.dot(theta.T))  
    for it in range(0, h.shape[0]):  
        if h[it]>0.5:  
            p[it,0]=1  
        else:  
            p[it,0]=0
    count = 0
    for i in range(0,100):
        if p[i] == y[i]:
            count += 1
    # Computing accuracy
    print '+++++++++++++++++++++++++++'
    print 'Train Accuracy: ', count/100.
    print '+++++++++++++++++++++++++++'
    # plot the final results
    print 'plot the classifier obtained'
    pos = where(y == 1)
    neg = where(y == 0)
    plt.figure()
    plt.plot(X[pos, 1], X[pos, 2], marker='o', c='b')
    plt.plot(X[neg, 1], X[neg, 2], marker='o', c='r')
    plt.plot([0.5, (-theta[0]-0.5*theta[1])/theta[2]], [(-theta[0]-0.5*theta[2])/theta[1], 0.5])
    plt.xlabel('feature1 (X1)')
    plt.ylabel('feature2 (X2)')
    plt.show()


# claim global variable
theta = np.ones(3)  # weight
theta[0] = -10.
alpha = 1.0  # learning rate
print 'The learning rate is given as:', alpha


if __name__ == '__main__':
    # load the data
    X, y = loadData()
    # train the model
    theta = train(theta, X, y, alpha)
    # predict the results and calculate the accuracy
    predict(theta, X, y)

