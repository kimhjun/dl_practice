import preprop_data
import simple_dnn as dnn
import numpy as np
import matplotlib.pyplot as plt

class SimpleDNNModel(object):
    def __init__(self):
        self.layers = [
            dnn.MiddleLayer(img_size * img_size, n_mid),
            dnn.MiddleLayer(n_mid, n_mid),
            dnn.OutputLayer(n_mid, n_out)
        ]

    def forward_propagation(self, x):
        for layer in self.layers:
            layer.forward(x)
            x = layer.y
        return x

    def backward_propagation(self, t):
        grad_y = t
        for layer in reversed(self.layers):
            layer.backward(grad_y)
            grad_y = layer.grad_x
        return grad_y

    def update_params(self, eta):
        for layer in self.layers:
            layer.update(eta)

    def get_error(self, x, t):
        y = self.forward_propagation(x)
        return -np.sum(t*np.log(y+1e-7)) / len(y)

    def get_accuracy(self, x, t):
        y = self.forward_propagation(x)
        count = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1))
        return count / len(y)

if __name__ == "__main__":
    img_size= 8
    n_mid = 16
    n_out = 10
    eta = 0.001
    epochs = 51
    batch_size = 32
    interval = 5
    
    DDL = preprop_data.DigitDataLoader(n_out)
    x_train, x_test, t_train, t_test = DDL.get_train_test_digits_data()

    simplednn = SimpleDNNModel()

    error_record_train = []
    error_record_test = []

    n_batch = len(x_train) // batch_size

    for i in range(epochs):
        index_random = np.arange(len(x_train))
        np.random.shuffle(index_random)

        for j in range(n_batch):
            mb_index = index_random[j*batch_size: (j+1)*batch_size]
            x_mb = x_train[mb_index, :]
            t_mb = t_train[mb_index, :]

            simplednn.forward_propagation(x_mb)
            simplednn.backward_propagation(t_mb)

            simplednn.update_params(eta)

        error_train = simplednn.get_error(x_train, t_train)
        error_record_train.append(error_train)
        error_test = simplednn.get_error(x_test, t_test)
        error_record_test.append(error_test)

        if i % interval == 0:
            print(f"Epoch: {str(i+1)}/{str(epochs)} Error_train: {error_train:.3f} Error_test: {error_test:.3f}")

    plt.plot(range(1, len(error_record_train)+1), error_record_train, label="Train")
    plt.plot(range(1, len(error_record_test)+1), error_record_test, label="Test")
    plt.legend()

    plt.xlabel("Epochs")
    plt.ylabel("Error")
    plt.savefig("traintest_error.png", dpi=300)

    acc_train = simplednn.get_accuracy(x_train, t_train)
    acc_test = simplednn.get_accuracy(x_test, t_test)
    print(f"Acc_train: {acc_train*100:.2f}%")
    print(f"Acc_test: {acc_test*100:.2f}%")