"""

    Created on 13/01/21 12:14 AM 
    @author: Kartik Prabhu

"""
import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        # Initialized with random numbers from a gaussian N(0, 0.001)
        self.weight = np.matlib.randn(input_dim, output_dim) * 0.001
        self.bias = np.matlib.randn((1, output_dim)) * 0.001
        self.gradWeight = np.zeros_like(self.weight)
        self.gradBias = np.zeros_like(self.bias)

    # y = mx + c
    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def getParameters(self):
        return [self.weight, self.bias]

    def backward(self, x, outputGrad):
        self.gradWeight = np.dot(x.T, outputGrad)
        self.gradBias = np.copy(outputGrad)
        return np.dot(outputGrad, self.weight.T)

class Activation:
    def forward(self,x):
        return 0

    """
    During back propogation we just multiply the gradient of the output, with gradient of the current function.
    """
    def backward(self,x,outputGrad):
        return 0

class Sigmoid(Activation):
    def forward(self, x):
        return 1/(1+np.exp(-x))

    def backward(self,x,outputGrad):
        sig = self.forward(x)
        return outputGrad * sig * (1 - sig)

class Relu(Activation):
    def forward(self, x):
        return np.maximum(0,x)

    def backward(self,x,outputGrad):
        dZ = np.array(outputGrad, copy = True)
        dZ[x <= 0] = 0
        return dZ

class Tanh(Activation):
    def forward(self,x):
        return np.tanh(x)

    def backward(self,x,outputGrad):
        output = np.tanh(x)
        return (1.0 - np.square(output)) * outputGrad

class Softmax(Activation):

    def forward(self,x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backward(self,predicted_output,ground_truth):
        grad = predicted_output
        for i,l in enumerate(ground_truth): #check if the value in the index is 1 or not, if yes then take the same index value from the predicted_ouput list and subtract 1 from it.
             if l == 1:
                grad[i] -= 1
        return grad