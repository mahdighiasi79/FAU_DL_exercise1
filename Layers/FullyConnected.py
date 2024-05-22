import Base
import numpy as np


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, input_size * output_size).reshape((input_size, output_size))
        self.biases = np.zeros(output_size)

    def forward(self, input_tensor):
        output_tensor = input_tensor @ self.weights
        output_tensor += self.biases
        return output_tensor
