# About neural network 

## basic structure

neural network = input_layer + n*(hidden layer + active function) + output_layer 

![network](network.png "Magic Gardens")

The input size is based on yout input data's seize. For example, if you input a $28 \times 28$ pixels' picture with 1 channel, the size of input is 7841 ($28 \times 28 \times 1$) which the input layer has 784 neurals. The number of hidden layer can define by yourself. The number of neural(N<sub>neural</sub>) in each hidden layer can also define by yourself. You can try different number of neruals and layers(N<sub>layer</sub>) to optimize your nerwork's performance. Keep in mind that if the N<sub>neural_total</sub> is too small means there have less space to help the loss function be decrease which might cause underfitting. higher N<sub>neural</sub> will cost higher computer resources. 

If N<sub>neural</sub> = 0: the network only can do linearly separable stuff
If N<sub>neural</sub> = 1: Can approximate any function that contains a continuous mapping from one finite space to another.
If N<sub>neural</sub> = 2:
If N<sub>neural</sub> > 2:

The output size is depend on what you want to know via this neural network. For example, if you want to know the output data is 'a' or 'b', your output value can be set as a $1 \times 2$ array that [0 1] represent as 'a', [1 0 ] represent as 'b'. Normally, the N<sub>output_layer</sub> = 1



## forward

## back p


## diff between neural network & deep neural network