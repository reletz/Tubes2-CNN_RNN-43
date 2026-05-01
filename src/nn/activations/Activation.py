import numpy as np

# NB: upstream = hasil chain derivation

class Activation:
    """Base class (meant to be ABC)"""
    def forward(self, x):
        """
        Args:
            x: Input array of shape (batch_size, ...)
            
        Returns:
            Output array of same shape
        """
        raise NotImplementedError("Subclasses must implement forward()")
    
    def backward(self, upstream_grad):
        """ Gradient calculation
        
        Args:
            upstream_grad: Gradient from next layer of shape (batch_size, ...)
            
        Returns:
            Gradient with respect to input of same shape
        """
        raise NotImplementedError("Subclasses must implement backward()")

class Linear(Activation):
    def forward(self, x):
        self.input = x
        self.output = x
        return self.output
    
    def backward(self, upstream_grad): # upstream * dx/dx = upstream * 1
        return upstream_grad

class ReLU(Activation):
    """Rectified Linear Unit"""
    def forward(self, x): # no negative
        self.input = x
        self.output = np.maximum(0, x)
        return self.output
    
    def backward(self, upstream_grad):
        mask = (self.input > 0).astype(float) # 0 or 1
        return upstream_grad * mask

class Sigmoid(Activation):
    def forward(self, x):
        self.input = x
        self.output = 1 / (1 + np.exp(-x))
        return self.output
    
    def backward(self, upstream_grad):
        return upstream_grad * self.output * (1 - self.output)

class Tanh(Activation):
    """Hyperbolic tangent"""
    def forward(self, x):
        self.input = x
        self.output = np.tanh(x)
        return self.output
    
    def backward(self, upstream_grad): # 1 - tanh^2
        return upstream_grad * (1 - self.output ** 2)

class Softmax(Activation):
    """For multi-class classification"""
    def forward(self, x):
        """
        gabisa lgsg implement; e^[some big number] bakal error
        softmax behaviour: hasilnya ga berubah kalau semua input dikurangin konstanta yang sama.
        implementasi: kurangin sama angka terbesar, baru kalkulasi outputnya
        """
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        self.input = x
        return self.output
    
    def backward(self, upstream_grad):
        """Use softmax Jacobian"""
        s = self.output
        ds = upstream_grad * s
        grad_input = ds - s * np.sum(ds, axis=1, keepdims=True)
        return grad_input

# Bonus

class LeakyReLU(Activation):
    """Leaky Rectified Linear Unit.
    
    f(x) = x if x > 0, else alpha * x (default alpha=0.01)
    Derivative: 1 if x > 0, else alpha
    """
    def __init__(self, alpha=0.01):
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        self.output = np.maximum(x, 0) + self.alpha * np.minimum(x, 0)
        return self.output
    
    def backward(self, upstream_grad):
        """Gradient: 1 if x > 0, else alpha"""
        mask = (self.input > 0).astype(float)
        return upstream_grad * (mask + self.alpha * (1 - mask))
    
class ELU(Activation):
    """Exponential Linear Unit.
    
    f(x) = x if x > 0, else alpha * (exp(x) - 1) (default alpha=1.0)
    Derivative: 1 if x > 0, else alpha * exp(x)
    """
    def __init__(self, alpha=1.0):
        self.alpha = alpha
    
    def forward(self, x):
        self.input = x
        self.output = np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
        return self.output
    
    def backward(self, upstream_grad):
        """Gradient: 1 if x > 0, else alpha * exp(x)"""
        exp_x = np.exp(self.input)
        mask = (self.input > 0).astype(float)
        return upstream_grad * (mask + self.alpha * exp_x * (1 - mask))
    
"""
https://datascience.stackexchange.com/questions/102483/relu-vs-leaky-relu-vs-elu-with-pros-and-cons

ELU
ELU is very similiar to RELU except negative inputs. 
They are both in identity function form for non-negative inputs. 
On the other hand, ELU becomes smooth slowly until its output equal to -alpha whereas RELU sharply smoothes.

Pros

ELU becomes smooth slowly until its output equal to -alpha whereas RELU sharply smoothes.
ELU is a strong alternative to ReLU.
Unlike to ReLU, ELU can produce negative outputs.

Cons

For x>0, it can blow up the activation with the output range of [0, inf].

ReLU
Pros

It avoids and rectifies vanishing gradient problem.
ReLu is less computationally expensive than tanh and sigmoid because it involves simpler mathematical operations.

Cons

One of its limitations is that it should only be used within hidden layers of a neural network model.
Some gradients can be fragile during training and can die. It can cause a weight update which will makes it never activate on any data point again. In other words, ReLu can result in dead neurons.
In another words, For activations in the region (x<0) of ReLu, 
gradient will be 0 because of which the weights will not get adjusted during descent.
That means, those neurons which go into that state will stop responding to variations in error/ input (simply because gradient is 0, nothing changes). This is called the dying ReLu problem.
The range of ReLu is [0,∞). This means it can blow up the activation.

LeakyRelu
LeakyRelu is a variant of ReLU. 
Instead of being 0 when z<0, a leaky ReLU allows a small, non-zero, constant gradient alpha (Normally, alpha=0.01). 
However, the consistency of the benefit across tasks is presently unclear. [1]

Pros

Leaky ReLUs are one attempt to fix the “dying ReLU” problem by having a small negative slope (of 0.01, or so).
Cons

As it possess linearity, it can't be used for the complex Classification. 
It lags behind the Sigmoid and Tanh for some of the use cases.
"""