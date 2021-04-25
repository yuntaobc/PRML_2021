import numpy as np


def sigmoid(inputs):
    result = 1 / (1 + np.exp(-inputs))
    return result


class Layer:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.rand(output_dim, input_dim)
        self.bias = np.random.rand(output_dim)
        self.result = 0

    def call(self, inputs):
        if not self.check(inputs):
            return False
        self.result = inputs.dot(self.weights.T) + self.bias
        self.result = sigmoid(self.result)
        return self.result

    def check(self, data):
        if type(data) != np.ndarray:
            print(f"Input data type is {type(data)} should be numpy.ndarray")
            return False
        if data.shape.__len__() != 2:
            print(f"Input data dim is {data.shape.__len__()} should be 2")
            return False
        if data.shape[1] != self.input_dim:
            print(f"Input data shape is {data.shape[1]} should be (n, {self.input_dim})")
            return False
        return True
