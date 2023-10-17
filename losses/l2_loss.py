import numpy as np

class L2Loss:
    def __init__(self) -> None:
        pass
    
    def forward(self,y_out, y_target):
        batch_size = y_out.shape[0]
        return np.sum(
            np.power(y_out-y_target ,2)
        ) / batch_size
    
    def backward(self, y_out , y_target):
        return np.multiply(2, y_out - y_target)
    