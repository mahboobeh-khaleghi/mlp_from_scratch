import numpy as np

class CrossEntropy:
    def __init__(self) -> None:
        pass
    
    def forward(self, y_out , y_target):
        batch_size = y_out.shape[0]
        return - np.sum(
            np.multiply(y_target ,np.log(y_out))
        ) / batch_size
        
    def backward(self,y_out , y_target):
        return -np.divide(y_target , y_out)