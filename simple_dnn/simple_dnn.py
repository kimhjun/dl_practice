import numpy as np

class BaseLayer(object):
    def update(self, eta):
        # eta: learning rate
        # update weight after backpropagation
        self.w -= eta * self.grad_w 
        self.b -= eta * self.grad_b

class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper) # He Initialization - using ReLU as activation func
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b # Linear function - from input to estimate result (y_hat)
        self.y = np.where(self.u <= 0, 0, self.u) # ReLU

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) # Derivative of ReLU
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

class OutputLayer(BaseLayer):
    def __init__(self, n_upper, n):
        # Xavier Initialization
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.exp(u)/np.sum(np.exp(u), axis=1, keepdims=True)
    
    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

def forward_propagation(x):
    for layer in layers:
        layer.forward(x)
        x = layer.y
    return x

def backward_propagation(t):
    grad_y = t
    for layer in reversed(layers):
        layer.backward(grad_y)
        grad_y = layer.grad_x
    return grad_y

def update_params(eta):
    for layer in layers:
        layer.update(eta)

def get_error(x, t):
    y = forward_propagation(x)
    return -np.sum(t*np.log(y+1e-7)) / len(y)

def get_accuracy(x, t):
    y = forward_propagation(x)
    count = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))
    return count / len(y)