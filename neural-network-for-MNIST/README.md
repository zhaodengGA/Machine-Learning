  ##       MNIST Single Layer Neural Network 

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
