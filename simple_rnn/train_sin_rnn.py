import numpy as np
import matplotlib.pyplot as plt

import simple_rnn
import gen_sinusoid_data

class SinusoidRNNTrainer(object):
    def __init__(self, n_time=10, n_in=1, n_mid=20, n_out=1, eta=0.001, epochs=51, batch_size=8, interval=5):
        self.n_time = n_time
        self.n_in = n_in
        self.n_mid = n_mid
        self.n_out = n_out

        self.eta = eta
        self.epochs = epochs
        self.batch_size = batch_size
        self.interval = interval
        self.rnn_layer, self.output_layer = self._set_model()
     

    def _load_data(self):
        input_data, correct_data = gen_sinusoid_data.gen_sin_data(self.n_time, self.n_in, self.n_out)
        return input_data, correct_data

    def _set_model(self):
        rnn_layer = simple_rnn.SimpleRNNLayer(self.n_in, self.n_mid)
        output_layer = simple_rnn.OutputLayer(self.n_mid, self.n_out)
        return rnn_layer, output_layer
    
    def train(self, x_mb, t_mb):
        y_rnn = np.zeros((len(x_mb), self.n_time+1, self.n_mid))
        y_prev = y_rnn[:, 0, :]

        
        for i in range(self.n_time):
            x = x_mb[:, i, :]
            self.rnn_layer.forward(x, y_prev)
            y = self.rnn_layer.y #rnn output
            y_rnn[:, i+1, :] = y
            y_prev = y
        
        self.output_layer.forward(y)

        self.output_layer.backward(t_mb)
        grad_y = self.output_layer.grad_x

        self.rnn_layer.reset_sum_grad()
        for i in reversed(range(self.n_time)):
            x = x_mb[:, i, :]
            y = y_rnn[:, i+1, :]
            y_prev = y_rnn[:, i, :]
            self.rnn_layer.backward(x, y, y_prev, grad_y)
        
        self.rnn_layer.update(self.eta)
        self.output_layer.update(self.eta)
    
    def predict(self, x_mb):
        y_prev = np.zeros((len(x_mb), self.n_mid))
        for i in range(self.n_time):
            x = x_mb[:, i, :]
            self.rnn_layer.forward(x, y_prev)
            y= self.rnn_layer.y
            y_prev = y
        
        self.output_layer.forward(y)
        return self.output_layer.y
    
    def get_error(self, x, t):
        y = self.predict(x)
        return 1.0/2.0*np.sum(np.square(y-t))

def main():
    trainer = SinusoidRNNTrainer()
    error_record = []
    input_data, correct_data = trainer._load_data()
    n_batch = len(input_data) // trainer.batch_size

    for i in range(trainer.epochs):
        index_random = np.arange(len(input_data))
        np.random.shuffle(index_random)

        for j in range(n_batch):
            mb_index = index_random[j*trainer.batch_size:(j+1)*trainer.batch_size]
            x_mb = input_data[mb_index]
            t_mb = correct_data[mb_index]
            trainer.train(x_mb, t_mb)
    
        error = trainer.get_error(input_data, correct_data)
        error_record.append(error)

        if i % trainer.interval == 0:
            print(f"Epoch: {int(i+1)}/{int(trainer.epochs)}, Error: {error:.3f}")
            predicted = input_data[0].reshape(-1).tolist()
            for i in range(40):
                x = np.array(predicted[-trainer.n_time:]).reshape(1, trainer.n_time, 1)
                y = trainer.predict(x)
                predicted.append(float(y[0, 0]))
                

if __name__ == "__main__":
    main()