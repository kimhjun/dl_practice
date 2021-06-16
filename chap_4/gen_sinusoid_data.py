import numpy as np

def gen_sin_data(n_time, n_in, n_out):
    sin_x = np.linspace(-2*np.pi, 2*np.pi)
    sin_y = np.sin(sin_x) + 0.1*np.random.randn(len(sin_x))

    n_sample = len(sin_x) - n_time
    input_data = np.zeros((n_sample, n_time, n_in))
    correct_data = np.zeros((n_sample, n_out))

    for i in range(0, n_sample):
        input_data[i] = sin_y[i:i+n_time].reshape(-1, 1)
        correct_data[i] = sin_y[i+n_time:i+n_time+1]
    return input_data, correct_data