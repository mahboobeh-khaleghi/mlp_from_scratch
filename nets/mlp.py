# from .linear_layer import LinearLayer
from nets import LinearLayer
# from activations import *
from nets import activations

class MLP:
    def __init__ (self,n_nodes, act_func_list, lr, l2_reg_coef):
        assert len(n_nodes) == len(act_func_list) + 1, "Error! The length of nodes list must be only one more than the length of activation functions list."
        
        self.n_nodes = n_nodes
        self.act_func_list = self.get_act_funcs(act_func_list)
        self.lr = lr
        self.l2_reg_coef = l2_reg_coef
        self.weight_decay = self.lr * self.l2_reg_coef
        self.layers = self.get_layers()
        
    def get_act_funcs(self, act_func_list):
        act_funcs = list()
        for act_func in act_func_list:
            if act_func.lower() == "relu":
                act_func = activations.ReLU()
            elif act_func.lower() == "leakyrelu":
                act_func = activations.LeakyReLU()    
            elif act_func.lower() == "softmax":
                act_func = activations.Softmax()
            elif act_func.lower() == "sigmoid":
                act_func = activations.Sigmoid()
            elif act_func.lower() == "tanh":
                act_func = activations.tanh()
            elif act_func.lower() == "identity":
                act_func = activations.Identity()
            else:
                print(f"EROR! undefined activation function -> {act_func}")
                exit()
                
            act_funcs.append(act_func)
            
        return act_funcs
    
    def get_layers(self):
        layers = []
        for i in range(len(self.act_func_list)):
            in_dim = self.n_nodes[i]
            out_dim = self.n_nodes[i+1]
            act_func = self.act_func_list[i]
            
            layer = LinearLayer(in_dim ,out_dim,act_func)
            layers.append(layer)
                
        return layers
   
    def forward(self,x):
        y_current = x
        for i in range (len(self.layers) ):
           y_next = self.layers[i].forward(y_current)
           y_current = y_next
           
        return y_current
    
    def backward(self, dloss_dy, x):
        dloss_da = dloss_dy
        
        for i in reversed(range(len(self.act_func_list))):
            
            if i-1<0:
                a_prev = x
            else:    
                a_prev = self.layers[i-1].activated
            
            dloss_dw, dloss_da_prev = self.layers[i].backward(dloss_da , a_prev)
            dloss_da = dloss_da_prev
            
            self.layers[i].update(dloss_dw, self.lr, self.weight_decay)
    
    def __repr__ (self):
        mess = ''
        for i in range (len(self.act_func_list)):
            # self.layers[i]
            # mess= mess + f"{i}: LinearLayer(in_dim={self.n_nodes[i]}, out_dim={self.n_nodes[i+1]},activation={self.act_func_list[i]})\n"
            mess = mess + f"{i}: {self.layers[i]}\n"
        return mess
    
    