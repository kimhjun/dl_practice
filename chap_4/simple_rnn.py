import numpy as np

class SimpleRNNLayer(object):
    def __init__(self, n_upper, n): # n_upper -> m
        # Xavier Initialization
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper) 
        self.v = np.random.randn(n, n) / np.sqrt(n)
        self.b = np.zeros(n)
    
    def forward(self, x, y_prev):
        u = np.dot(x, self.w) + np.dot(y_prev, self.v) + self.b
        self.y = np.tanh(u) # activation with tanh

    def backward(self, x, y, y_prev, grad_y):
        delta = grad_y * (1 - y**2)

        self.grad_w += np.dot(x.T, delta)
        self.grad_v += np.dot(y_prev.T, delta)
        self.grad_b += np.sum(delta, axis=0)

        self.grad_x = np.dot(delta, self.w.T)
        self.grad_y_prev = np.dot(delta, self.v.T)

    def reset_sum_grad(self):
        # reset cumulated grad_w before backward operation
        self.grad_w = np.zeros_like(self.w)
        self.grad_v = np.zeros_like(self.v)
        self.grad_b = np.zeros_like(self.b)

    def update(self, eta):
        self.w -= eta * self.grad_w
        self.v -= eta * self.grad_v
        self.b -= eta * self.grad_b


class OutputLayer(object): #FullyConnectedLayer
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)

    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = u
    
    def backward(self, t):
        delta = self.y - t

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b
    
    