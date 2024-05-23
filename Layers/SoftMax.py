import Base
import numpy as np


class SoftMax(Base.BaseLayer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_tensor):
        exp = np.power(np.e, input_tensor)
        s = np.sum(exp, keepdims=False)
        output_tensor = exp / s
        return output_tensor

    def backward(self, error_tensor):
        pass
