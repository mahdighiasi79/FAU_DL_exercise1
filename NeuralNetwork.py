import copy
import Optimization


class NeuralNetwork:

    def __init__(self):
        self.optimizer = Optimization.Optimizers.Sgd(0.1)
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None

    def forward(self, input_tensor, label_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)
        output_tensor = self.loss_layer.forward(input_tensor, label_tensor)
        return output_tensor

    def backward(self, label_tensor):
        error_tensor = self.loss_layer.backward(label_tensor)
        for layer in self.layers:
            error_tensor = layer.backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizer = copy.deepcopy(self.optimizer)
            layer.set_optimizer(optimizer)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            input_tensor, label_tensor = self.data_layer.next()
            self.forward(copy.deepcopy(input_tensor), label_tensor)
            error_tensor = self.backward(label_tensor)
            self.loss.append(error_tensor)

    def test(self, input_tensor):
        output_tensor = copy.deepcopy(input_tensor)
        for layer in self.layers:
            output_tensor = layer.forward(output_tensor)
        return output_tensor
