from sklearn.model_selection import train_test_split
import numpy as np
from sklearn import datasets

class DigitDataLoader(object):
    def __init__(self, n_out=10):
        self.digits_data = datasets.load_digits()
        self.n_out = n_out
    
    def normalize(self):
        input_data = np.asarray(self.digits_data.data)  #(1797 X 64)
        input_data = (input_data - np.average(input_data)) / np.std(input_data)
        return input_data

    def set_target_array(self):
        correct = np.asarray(self.digits_data.target)
        correct_data = np.zeros((len(correct), self.n_out))
        
        for i in range(len(correct)):
            correct_data[i, correct[i]] = 1    
        return correct_data
    
    def get_train_test_digits_data(self):
        input_data = self.normalize()
        correct_data = self.set_target_array()
        x_train, x_test, t_train, t_test = train_test_split(input_data, correct_data)
        return x_train, x_test, t_train, t_test