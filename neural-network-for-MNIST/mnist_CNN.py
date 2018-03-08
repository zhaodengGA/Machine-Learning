#! /usr/bin/python
# _*_ coding: utf-8 _*_
#
# ----------------------
# @Author: Zhao Deng
# ----------------------
#
# File name: mnist_CNN.py

'''
            ** Convolutional Neural Network for MNIST **

This code is a rewritten version of MNIST of Google Developers Codelabs for learning purposes. Here it shows how to build
and train a four layers Convolutional Neural Network that recognizes handwritten digits of MNIST dataset which has a
collection of 60,000 labeled digits. MNIST dataset could be downloaded at: http://yann.lecun.com/exdb/mnist/

* This is a four layers Convolutional Neural Network.
  The first two layers are built as Convolutional Neural Network.
  ReLU function is applied to the third layer as activation function.
  Softmax function is applied to the last layer as activation function.

* Tensorflow is applied to implement the algorithm.

* The data are loaded with the official tensorflow MNIST loader .

* Cross entropy has been chosen as Cost Function. 

* The Adam optimizer is chosen to train the weights {Wi} and the biases {bi}.

* The model is trained each time with 100 images in the minibatch.

* Test the model once with the 10,000 images in the test dataset after every 100 times of training.


# The structure of the neural network is:
------------------------------------------------------------------------------------------------------------------
layer 1:    X1[n_batch, 28, 28, 1]       W1[6, 6, 1, 4]     b1[4]    Y1[n_batch, 28, 28, 4]   stride=1   <CNN>
layer 2:    X2=Y1=[n_batch, 28, 28, 4]   W1[6, 6, 4, 10]    b1[10]   Y2[n_batch, 14, 14, 10]  stride=2   <CNN>
layer 3:    X3=Y2=[n_batch, 14*14*10]    W1[14*14*10, 200]  b1[200]  Y3[n_batch, 200]                    <Relu>
layer 4:    X4=Y3=[n_batch, 200]         W1[200, 10]        b1[10]   Y4[n_batch, 10]                     <softmax>
------------------------------------------------------------------------------------------------------------------

The final test accuracy of this model reaches to 99%.

'''

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data


# assignment of coefficient
length = 28              # image length of each input image
n_batch = 100            # number of images in each minibatch
N1 = 6                   # the size of the filters of the 1st layer
C1 = 4                   # number of channel in the 1st layer
stride1 = 1              # the stride of the 1st layer
N2 = 6                   # the size of the filters of the 2nd layer
C2 = 10                  # number of channel in the 2nd layer
stride2 = 2              # the stride of the 2nd layer
L3 = 200                 # number of neurons in 3rd layer
L4 = 10                  # number of neurons in 4th layer


# load MNIST data with the default tensorflow mnist reader
mnist = input_data.read_data_sets('../data', one_hot=True, reshape=False)


# Define variables
X = tf.placeholder(tf.float32, ([None, length, length, 1]))
Y_tar = tf.placeholder(tf.float32, [None, L4])
learning_rate = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.truncated_normal([N1, N1 ,1 ,C1], stddev=0.1))
b1 = tf.Variable(tf.zeros([C1]))
W2 = tf.Variable(tf.truncated_normal([N2, N2 ,C1 ,C2], stddev=0.1))
b2 = tf.Variable(tf.zeros([C2]))
W3 = tf.Variable(tf.truncated_normal([length/stride2*length/stride2*C2, L3], stddev=0.1))
b3 = tf.Variable(tf.zeros([L3]))
W4 = tf.Variable(tf.truncated_normal([L3, L4], stddev=0.1))
b4 = tf.Variable(tf.zeros([L4]))


# The first two layers are built as Convolutional Neural Network
# The data are treated by Relu function after convolution operation
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, [1, stride1, stride1, 1], padding='SAME') + b1)
Y2 = tf.nn.relu(tf.nn.conv2d(Y1, W2, [1, stride2, stride2, 1], padding='SAME') + b2)


# ReLU function is applied to the third layer as activation function
Y2_2D = tf.reshape(Y2, [-1, length/stride2*length/stride2*C2])
Y3 = tf.nn.relu(tf.matmul(Y2_2D, W3) + b3)


# Softmax function is applied to the last layer as activation function
Ylogits = tf.matmul(Y3, W4) + b4
Y4 = tf.nn.softmax(Ylogits)


# Define the accuracy
correct_predict = tf.equal(tf.argmax(Y4, 1), tf.argmax(Y_tar, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, tf.float32))


# Cross entropy has been chosen as Cost Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_tar)
cross_entropy = tf.reduce_mean(cross_entropy)*100.


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

    # Learning rate decay process
    max_lr = 0.003
    min_lr = 0.0001
    norm = 2000.
    lr_i = min_lr + (max_lr - min_lr)*np.exp(-i/norm)

    if i%10 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: X_batch, Y_tar: Y_batch})
        print i, ': accuracy is:', a, ' cross entropy is:', c, ' learning rate is:', lr_i

    if i%100 == 0:
        a, c = sess.run([accuracy, cross_entropy], feed_dict={X: mnist.test.images, Y_tar: mnist.test.labels})
        print '******* epoch',i/100,'******* The test accuracy is:',a,' cross entropy is:', c
        if i == 2000:
            print ''
            print '+++++++++++++++++++++++++++++++++++'
            print 'The last test accuracy is:', a
            print '+++++++++++++++++++++++++++++++++++'

    sess.run(train, feed_dict={X: X_batch, Y_tar: Y_batch, learning_rate: lr_i})


if __name__ == '__main__':
    for i in range(2001):
        training(i)
    print 'The end.'
