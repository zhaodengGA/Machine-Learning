#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: mnist_single_layer_NN.py

'''
            ** MNIST Single Layer Neural Network **

This code is a rewritten version of Google Developers Codelabs for learning purposes. This code shows how to build
and train a single layer neural network that recognises handwritten digits of MNIST dataset which has a collection of
60,000 labeled digits. MNIST dataset could be downloaded at: http://yann.lecun.com/exdb/mnist/

* Tensorflow is applied to implement the algorithm.

* Softmax function is applied as activation function. 

* Cross entropy has been chosen as Cost Function. 

* The data are loaded with the official tensorflow MNIST loader .

* The gradient descent method is applied to train the weight theta.

* The model is trained each time with 100 images in the minibatch.

* Test the model once with the 10,000 images in the test dataset after every 100 times of training.

The final test accuracy of this model reaches to 92%.

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# assignment of coefficient
learning_rate = 0.005   # learning rate for gradient descent method
length = 28             # image length of each input image
n_neu = 10              # number of neurons in the network
n_batch = 100           # number of images in each minibatch


# load MNIST data with the default tensorflow mnist reader
mnist = input_data.read_data_sets('../data', one_hot=True, reshape=False)


X = tf.placeholder(tf.float32, [None, length, length, 1])  # images of 28x28 as inputs
Y_target = tf.placeholder(tf.float32, [None, n_neu])       # the target result (the actual number) of each image
W = tf.Variable(tf.ones([length**2, n_neu]))               # weights
b = tf.Variable(tf.zeros([n_neu]))                         # biases

X_2D = tf.reshape(X, [-1, length**2])
# Softmax function is applied as activation function
Y = tf.nn.softmax(tf.matmul(X_2D, W) + b)


# Cross entropy has been chosen as Cost Function
# Pay attention that the entropy should be defined as multiplying the matrix element by element.
# So don't write cross entropy as: cross_entropy = - tf.reduce_mean(tf.matmul(tf.transpose(Y_target), tf.log(Y)))
cross_entropy = -tf.reduce_mean(Y_target * tf.log(Y)) * 1000.0
index_Y = tf.argmax(Y, 1)
index_Y_target = tf.argmax(Y_target, 1)
right_prediction = tf.equal(index_Y, index_Y_target)
accuracy = tf.reduce_mean(tf.cast(right_prediction, tf.float32))
# The gradient descent method is applied to train the weight theta
gradient_descent = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)


# initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# The model is trained each time with n_batch images in the minibatch
def training(i):
    X_batch, Y_batch = mnist.train.next_batch(n_batch)

    if i%10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: X_batch, Y_target: Y_batch})
        print i, ': accuracy is:', a, ' cross entropy is:', c

    if i%100 == 0:
        # Test the model once with the 10,000 images in the test dataset after every 100 times of training
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_target: mnist.test.labels})
        print '******* epoch',i/100,'******* The test accuracy is:',a,' cross entropy is:', c
        if i == 2000:
            print ''
            print '+++++++++++++++++++++++++++++++++++'
            print 'The last test accuracy is:', a
            print '+++++++++++++++++++++++++++++++++++'

    # Train the model with gradient descent method
    sess.run(gradient_descent, feed_dict={X: X_batch, Y_target: Y_batch})


for i in range(2001):
    training(i)
print 'The end.'