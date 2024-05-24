import Base
import numpy as np


class FullyConnected(Base.BaseLayer):

    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.uniform(0, 1, input_size * output_size).reshape((input_size, output_size))
        # self.biases = np.zeros(output_size)
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor = np.array([])

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = input_tensor @ self.weights
        # output_tensor += self.biases
        return output_tensor

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, value):
        self._optimizer = value

    optimizer = property(get_optimizer, set_optimizer)

    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, value):
        self._gradient_weights = value

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def backward(self, error_tensor):
        gradient_weights = self.input_tensor.transpose() @ error_tensor
        self.gradient_weights = gradient_weights

        error_tensor = self.weights @ error_tensor.transpose()

        if self.optimizer is not None:
            self.optimizer.calculate_update(self.weights, gradient_weights)

        return error_tensor
