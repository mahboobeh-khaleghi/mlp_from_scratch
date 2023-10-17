import numpy as np

class ReLU:
    def __init__(self):
        pass
    def forward (self, x):
        """
            Y = ReLU(x)
            
            parameters:
                * x: input of relu
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = relu(x)
                    * size: (batch_size, in_dim) 
        """
        return np.maximum(0,x)
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of relu
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        dy = np.array(x, dtype=float)
        dy[dy>0]=1 
        dy[dy==0]=0.5
        dy[dy<0]=0
       
        return dy
    
    def __repr__(self):
        return f"ReLU"
    
class LeakyReLU:
    def __init__(self,alpha) -> None:
        self.alpha = alpha
        
    def forward (self, x):
        """
            Y = LeakyReLU(x)
            
            parameters:
                * x: input of leaky-relu
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = leaky-relu(x)
                    * size: (batch_size, in_dim) 
        """
        # return np.maximum(x , self.alpha*x)
        return np.maximum(x, np.multiply(self.alpha,x))         # Self.alpha * x does not calculate the multiplication accurately!!!!!!
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of leaky-relu
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        dy = np.array(x, dtype=float)
        dy[dy>0]=1 
        dy[dy==0]=(1+self.alpha)/2
        dy[dy<0]=self.alpha
       
        return dy
    
    def __repr__(self):
        return f"LeakyReLU"
    
class Sigmoid:
    def __init__(self) -> None:
        pass
    def forward (self, x):
        """
            Y = Sigmoid(x)
            
            parameters:
                * x: input of Sigmoid
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = Sigmoid(x)
                    * size: (batch_size, in_dim) 
        """
        
        return 1/(1+np.exp(-x))
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of Sigmoid
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        s=self.forward(x)
       
        return np.multiply(s,1-s)
    
    def __repr__(self):
        return f"Sigmoid"
    
class tanh:
    def __init__(self) -> None:
        pass
    def forward (self, x):
        """
            Y = tanh(x)
            
            parameters:
                * x: input of tanh
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = tanh(x)
                    * size: (batch_size, in_dim) 
        """
        
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of tanh
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        tanh=self.forward(x)
       
        return 1 - np.power(tanh, 2)
    
    def __repr__(self):
        return f"Tanh"
    
class Softmax:
    def __init__(self) -> None:
        pass
    def forward (self, x):
        """
            Y = Softmax(x)
            
            parameters:
                * x: input of Softmax
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = Softmax(x)
                    * size: (batch_size, in_dim) 
        """
        # shiftx = x - np.max(x, axis=1).reshape(x.shape[0],1)
        shiftx = x - np.max(x, axis=1)[:, np.newaxis]
        
        exps = np.exp(shiftx)
        # exps = np.exp(x)
        # print(f"exps: {exps}")
        
        return exps/ np.sum(exps , axis=1)[:,np.newaxis]
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of Softmax
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        softmax = self.forward(x)
        output = np.multiply(softmax , np.eye(x.shape[1]) - softmax.T)
        return output
    
    def __repr__(self):
        return f"Softmax"
    
class Identity:
    def __init__(self) -> None:
        pass
        
    def forward (self, x):
        """
            Y = Identity(x)
            
            parameters:
                * x: input of leaky-relu
                    * size: (batch_size, in_dim)
                    
            output:
                * y: y = leaky-relu(x)
                    * size: (batch_size, in_dim) 
        """
        return x         
    
    def backward (self, x):
        """
            dY/dx 
            
            parameters:
                * X: input of leaky-relu
                    * size: (batch_size, in_dim)
                    
            outputs:
                * dy: dy/dx
                    * size: (batch_size, in_dim)
        """
        dy = np.ones_like(x)
        # dy = np.ones((x.shape[0], x.shape[1]))
        # dy = np.ones(x.shape)
       
        return dy
    
    def __repr__(self):
        return f"Identity"
    