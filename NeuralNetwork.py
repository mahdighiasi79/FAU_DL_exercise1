import copy
import numpy as np
import Optimization


class NeuralNetwork:

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = copy.deepcopy(label_tensor)
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output_tensor = self.loss_layer.forward(input_tensor, label_tensor)
        return output_tensor

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        reversed_layers = copy.deepcopy(self.layers)
        reversed_layers.reverse()
        for layer in reversed_layers:
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.backward(self.label_tensor)
            self.loss.append(loss)

    def test(self, input_tensor):
        output_tensor = copy.deepcopy(input_tensor)
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor
