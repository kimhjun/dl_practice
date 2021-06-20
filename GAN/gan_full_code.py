import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

img_size = 8
n_noise = 16
eta = 0.001
n_learn = 10001
interval = 1000
batch_size = 32

digits_data = datasets.load_digits()
x_train = np.asarray(digits_data.data)
x_train = x_train / 15*2-1
t_train = digits_data.target

class BaseLayer(object):
    def update(self, eta):
        self.w -= eta * self.grad_w
        self.b -= eta * self.grad_b

class MiddleLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) * np.sqrt(2/n_upper)
        self.b = np.zeros(n)
    
    def forward(self, x):
        self.x = x
        self.u = np.dot(x, self.w) + self.b
        self.y = np.where(self.u <= 0, 0, self.u) #ReLU

    def backward(self, grad_y):
        delta = grad_y * np.where(self.u <= 0, 0, 1) # Derivative of Relu

        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

class GenOutLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = np.tanh(u) # activate with tanh at output layer
    
    def backward(self, grad_y):
        delta = grad_y * (1 - self.y**2)
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)

#output layer of discriminator
class DiscOutLayer(BaseLayer):
    def __init__(self, n_upper, n):
        self.w = np.random.randn(n_upper, n) / np.sqrt(n_upper)
        self.b = np.zeros(n)
    
    def forward(self, x):
        self.x = x
        u = np.dot(x, self.w) + self.b
        self.y = 1/(1+np.exp(-u)) #sigmoid activation

    def backward(self, t):
        delta = self.y - t
        self.grad_w = np.dot(self.x.T, delta)
        self.grad_b = np.sum(delta, axis=0)
        self.grad_x = np.dot(delta, self.w.T)
    

gen_layers = [
    MiddleLayer(n_noise, 32),
    MiddleLayer(32, 64),
    GenOutLayer(64, img_size*img_size)
]

disc_layers = [
    MiddleLayer(img_size*img_size, 64),
    MiddleLayer(64, 32),
    DiscOutLayer(32, 1)
]

def forward_propagation(x, layers):
    for layer in layers:
        layer.forward(x)
        x = layer.y
    return x

def backpropagation(t, layers):
    grad_y = t
    for layer in reversed(layers):
        layer.backward(grad_y)
        grad_y = layer.grad_x
    return grad_y

def update_params(layers):
    for layer in layers:
        layer.update(eta)

def get_error(y, t):
    eps = 1e-7

    return -np.sum((t*np.log(y+eps) + (1-t)*np.log(1-y+eps))) / len(y)

def get_accuracy(y, t):
    correct = np.sum(np.where(y < 0.5, 0, 1) == t)
    return correct / len(y)

def train_model(x, t, prop_layers, update_layers):
    y = forward_propagation(x, prop_layers)
    backpropagation(t, prop_layers)
    update_params(update_layers)
    return (get_error(y, t), get_accuracy(y, t))

def generate_images(i):
    n_rows = 16
    n_cols = 16
    noise = np.random.normal(0, 1, (n_rows*n_cols, n_noise))
    g_imgs = forward_propagation(noise, gen_layers)
    g_imgs = g_imgs/2 + 0.5
    print(noise.shape, g_imgs.shape)
    img_size_spaced = img_size + 2

    matrix_image = np.zeros((img_size_spaced*n_rows, img_size_spaced*n_cols))

    for r in range(n_rows):
        for c in range(n_cols):
            g_img = g_imgs[r*n_cols + c].reshape(img_size, img_size)
            top = r*img_size_spaced
            left = c*img_size_spaced
            matrix_image[top: top+img_size, left: left+img_size] = g_img

    plt.figure(figsize=(8, 8))
    plt.imshow(matrix_image.tolist(), cmap='Greys_r')

    plt.tick_params(labelbottom=False, labelleft=False, bottom=False, left=False)
    plt.show()

batch_half = batch_size // 2
error_record = np.zeros((n_learn, 2))
acc_record = np.zeros((n_learn, 2))

for i in range(n_learn):
    # train discriminator: noise in fake img out then train disc
    noise = np.random.normal(0, 1, (batch_half, n_noise))
    imgs_fake = forward_propagation(noise, gen_layers)
    t = np.zeros((batch_half, 1))
    error, accuracy = train_model(imgs_fake, t, disc_layers, disc_layers)
    error_record[i][0] = error
    acc_record[i][0] = accuracy

    # train disc with real img
    rand_ids = np.random.randint(len(x_train), size=batch_half)
    imgs_real = x_train[rand_ids, :]
    t = np.ones((batch_half, 1))
    error, accuracy = train_model(imgs_real, t, disc_layers, disc_layers)
    error_record[i][1] = error
    acc_record[i][1] = accuracy

    # conjugate disc and gen. train only gen
    noise = np.random.normal(0, 1, (batch_size, n_noise))
    t = np.ones((batch_size, 1))
    train_model(noise, t, gen_layers+disc_layers, gen_layers)

    if i % interval == 0:
        print(f"n_learn: {i}")
        print(f"Error_fake: {error_record[i][0]}, Acc_fake: {acc_record[i][0]}")
        print(f"Error_real: {error_record[i][1]}, Acc_real: {acc_record[i][1]}")
        generate_images(i)