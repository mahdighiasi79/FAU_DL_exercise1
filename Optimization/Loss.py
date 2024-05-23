import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        return

    @staticmethod
    def forward(prediction_tensor, label_tensor):
        loss = label_tensor * np.log(prediction_tensor)
        loss = -np.sum(loss, keepdims=False)
        return loss

    def backward(self, label_tensor):
        pass
