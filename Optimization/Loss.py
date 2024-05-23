import numpy as np


class CrossEntropyLoss:

    def __init__(self):
        self.prediction_tensor = np.array([])

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        loss = label_tensor * np.log(prediction_tensor)
        loss = -np.sum(loss, keepdims=False)
        return loss

    def backward(self, label_tensor):
        error_tensor = (-np.log(np.e) * label_tensor) / self.prediction_tensor
        return error_tensor
