import Base
import numpy as np


class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()
        self.output_tensor = np.array([])

    def forward(self, input_tensor):
        exp = np.power(np.e, input_tensor)
        s = np.sum(exp, keepdims=False)
        output_tensor = exp / s
        self.output_tensor = output_tensor
        return output_tensor

    def backward(self, error_tensor):
        softmax_derivative = self.output_tensor - np.power(self.output_tensor, 2)
        error_tensor *= softmax_derivative
        return error_tensor
