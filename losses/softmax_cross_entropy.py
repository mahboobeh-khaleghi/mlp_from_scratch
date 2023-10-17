import numpy as np 
from nets import activations

class SoftmaxCrossEntropy:
    def __init__(self) -> None:
        self.softmax_act = activations.Softmax()
    
    def forward(self, score , y_target,eps =1e-7):
        
        y_out = self.softmax_act.forward(score)
        ln_y_out = np.log (y_out+eps)
        loss_i = np.multiply(y_target, ln_y_out).sum(axis=1)
        
        return loss_i.sum(axis=0) / y_out.shape[0]
    
    def backward(self, score , y_target):
        # assert self.check_sum_y_target_one(y_target), "Error!!! Sum of Y_target in every data must equals to one."
        y_out = self.softmax_act.forward(score)
        return y_out - y_target
    
    def check_sum_y_target_one(self, y_target):
        batch_size = y_target.shape[0]
        
        # y_target = y_target.reshape(32,1)
        # does_one = np.sum(y_target, axis=1) == np.ones(batch_size)
        does_one = np.sum(y_target) == np.ones(batch_size)
        
        # print(y_target)
        # print(np.sum(y_target))
        # print(np.ones(batch_size))
        
        
        return np.sum(does_one) == batch_size
        
        