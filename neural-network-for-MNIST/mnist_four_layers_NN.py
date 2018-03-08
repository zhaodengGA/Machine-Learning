#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: mnist_four_layers_NN.py

'''
            ** MNIST four Layers Neural Network **

This code is a rewritten version of MNIST of Google Developers Codelabs for learning purposes. Here it shows how to build
and train a four layers neural network that recognizes handwritten digits of MNIST dataset which has a collection of 60,000
labeled digits. MNIST dataset could be downloaded at: http://yann.lecun.com/exdb/mnist/

* This is a four layers Neural Network.
  Sigmoid function is applied to the first three layers as activation function.
  Softmax function is applied to the last layer as activation function.

* Tensorflow is applied to implement the algorithm.

* The data are loaded with the official tensorflow MNIST loader .

* Cross entropy has been chosen as Cost Function. 

* The Adam optimizer is chosen to train the weights {Wi} and the biases {bi}.

* The model is trained each time with 100 images in the minibatch.

* Test the model once with the 10,000 images in the test dataset after every 100 times of training.


# The structure of the neural network is:
----------------------------------------------------------------------------------------------------
layer 1:    X1[n_batch, 28, 28, 1]    W1[28*28, 200]    b1[200]     Y1[n_batch, 200]      <sigmoid>
layer 2:    X2=Y1=[n_batch, 200]      W1[200, 50]       b1[50]      Y2[n_batch, 50]       <sigmoid>
layer 3:    X3=Y2=[n_batch, 50]       W1[50, 20]        b1[20]      Y3[n_batch, 20]       <sigmoid>
layer 4:    X4=Y3=[n_batch, 20]       W1[20, 10]        b1[10]      Y4[n_batch, 10]       <softmax>
----------------------------------------------------------------------------------------------------

The final test accuracy of this model reaches to 97%.

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# assignment of coefficient
learning_rate = 0.003   # learning rate for gradient descent method
length = 28             # image length of each input image
n_batch = 100           # number of images in each minibatch
L1 = 200                # number of neurons in 1st layer
L2 = 50                 # number of neurons in 2nd layer
L3 = 20                 # number of neurons in 3rd layer
L4 = 10                 # number of neurons in 4th layer


# Load MNIST data with the default tensorflow mnist reader
mnist = input_data.read_data_sets('../data', one_hot=True, reshape=False)


# Define variables
X = tf.placeholder(tf.float32, [None, length, length, 1])
Y_tar = tf.placeholder(tf.float32, [None, L4])
W1 = tf.Variable(tf.truncated_normal([length**2, L1], stddev=0.2))
b1 = tf.Variable(tf.zeros(L1))
W2 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.2))
b2 = tf.Variable(tf.zeros(L2))
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.2))
b3 = tf.Variable(tf.zeros(L3))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.2))
b4 = tf.Variable(tf.zeros(L4))


# Sigmoid function is applied to the first three layers as activation function
X_2D = tf.reshape(X, [-1, 784])
Y1 = tf.sigmoid(tf.matmul(X_2D, W1) + b1)
Y2 = tf.sigmoid(tf.matmul(Y1, W2) + b2)
Y3 = tf.sigmoid(tf.matmul(Y2, W3) + b3)
# Softmax function is applied to the last layer as activation function
Y_logits = tf.matmul(Y3, W4) + b4
Y4 = tf.nn.softmax(Y_logits)


# Cross entropy has been chosen as Cost Function
entropy_mat = tf.nn.softmax_cross_entropy_with_logits(logits=Y_logits, labels=Y_tar)
cross_entropy = tf.reduce_mean(entropy_mat)*100.


# Define the accuracy
correct_predict = tf.equal(tf.argmax(Y4,1), tf.argmax(Y_tar,1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# The Adam optimizer is chosen to train the weight theta
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(cross_entropy)


# Initialize the variables
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# Define the training process
def training(i):
    X_batch, Y_batch = mnist.train.next_batch(n_batch)

    if i%10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: X_batch, Y_tar: Y_batch})
        print i, ': accuracy is:', a, ' cross entropy is:', c

    if i%100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_tar: mnist.test.labels})
        print '******* epoch',i/100,'******* The test accuracy is:',a,' cross entropy is:', c
        if i == 2000:
            print ''
            print '+++++++++++++++++++++++++++++++++++'
            print 'The last test accuracy is:', a
            print '+++++++++++++++++++++++++++++++++++'

    sess.run(train, feed_dict={X: X_batch, Y_tar: Y_batch})


if __name__ == '__main__':
    for i in range(2001):
       training(i)
    print 'The end.'

