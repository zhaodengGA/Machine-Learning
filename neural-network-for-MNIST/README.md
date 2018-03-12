#                    Neural Network for MNIST  
<br>
	This code is capable to recognize the handwritten digits images. Three approaches have been listed here. They are single layer Neural Network, multiple layer Neural Network and Convolutional Neural Network. The corresponding recognition accuracy of these models reaches to 92%, 97%, 99% respectively. It shows the advance algorithm has better performance but more expensive. <br><br>
	This is a rewritten version of MNIST of Google Developers Codelabs for learning purposes. Here it shows how to build and train a Neural Network which recognizes handwritten digits of MNIST dataset which has a collection of 60,000 labeled digits. MNIST dataset could be downloaded at: http://yann.lecun.com/exdb/mnist/  <br><br>

Tensorflow is applied to implement the algorithm. The data are loaded with the official tensorflow MNIST loader. Cross entropy has been chosen as Cost Function. The Gradient Decent or Adam optimizer is chosen to train the weights {Wi} and the biases {bi}. The model is trained each time with 100 images in the minibatch. Test the model once with the 10,000 images in the test dataset after every 100 times of training. <br><br>

	
	

* The structure of the single layer neural network is:<br>
\-----------------------------------------------------<br>
input: 　X1[n_batch, 28, 28, 1]    <br>
layer 1: (10)  W1[28*28, 10]        b1[10]         Y1=[n_batch, 10]         <softmax><br>
\-----------------------------------------------------<br>


* The structure of the four layers neural network is:<br>
\-----------------------------------------------------<br>
layer 1: (200)    X1[n_batch, 28, 28, 1]    W1[28*28, 200]    b1[200]     Y1[n_batch, 200]      <sigmoid><br>
layer 2: (50)    　X2=Y1=[n_batch, 200]      W1[200, 50]       b1[50]      Y2[n_batch, 50]       <sigmoid><br>
layer 3: (20)    　X3=Y2=[n_batch, 50]       W1[50, 20]        b1[20]      Y3[n_batch, 20]       <sigmoid><br>
layer 4: (10)     　X4=Y3=[n_batch, 20]       W1[20, 10]        b1[10]      Y4[n_batch, 10]       <softmax><br>
\-----------------------------------------------------<br>
* The structure of the four layers convolutional neural network is:<br>
\-----------------------------------------------------<br>
	layer 1:    X1[n_batch, 28, 28, 1]       　　W1[6, 6, 1, 4]     　b1[4]    　Y1[n_batch, 28, 28, 4]   　stride=1   　<CNN><br>
	layer 2:    X2=Y1=[n_batch, 28, 28, 4]   　W1[6, 6, 4, 10]    　b1[10]   　Y2[n_batch, 14, 14, 10]  　stride=2   　<CNN><br>
	layer 3:    X3=Y2=[n_batch, 14*14*10]    　W1[14*14*10, 200]  　b1[200]  　Y3[n_batch, 200]                    　　<Relu><br>
	layer 4:    X4=Y3=[n_batch, 200]         　W1[200, 10]        　　b1[10]   　Y4[n_batch, 10]                     　　<softmax><br>
\-----------------------------------------------------<br>
	
