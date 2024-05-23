import Base


class ReLU(Base.BaseLayer):

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input_tensor):
        output_tensor = input_tensor > 0
        output_tensor *= input_tensor
        return output_tensor

    @staticmethod
    def backward(error_tensor):
        pass
