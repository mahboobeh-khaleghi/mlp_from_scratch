import numpy as np

class LinearLayer:
   
   def __init__(self,in_dim,out_dim, act_func) -> None:
      # self.weight = np.random.randn(out_dim , in_dim+1) * np.sqrt(2/( in_dim + out_dim ))
      lim = np.sqrt(2 / float(in_dim + out_dim))
      self.weight = np.random.normal(
         0.0, 
         lim, 
         size=(out_dim, in_dim+1)
      )
      # lim = np.sqrt(6 / float(in_dim))
      # self.weight = np.random.uniform(
      #    low = -lim,
      #    high = lim,
      #    size = (out_dim, in_dim+1)
      # )
      self.act_func = act_func
      self.in_dim = in_dim
      self.out_dim = out_dim
      self.score = np.zeros (out_dim)
      self.activated = np.zeros (out_dim)
      
      
   def forward (self, x):
      x_b = np.ones((x.shape[0], self.in_dim+1))
      x_b[: , :self.in_dim] = x
      
      self.score = np.dot(self.weight , x_b.T).T
      self.activated = self.act_func.forward(self.score)
      return self.activated
   
   def backward (self, dloss_da , activated_prev):
      activated_prev_b = np.ones((activated_prev.shape[0], activated_prev.shape[1]+1))
      activated_prev_b[: , :activated_prev.shape[1]] = activated_prev
      
      # print(f"self.score: {self.score.shape}")
      # print(f"self.activated: {self.activated.shape}")
      # print(f"dloss_da: {dloss_da.shape}")
      # print(f"activated_prev: {activated_prev.shape}")
      # activated_prev_b = np.ones((activated_prev.shape[0], activated_prev.shape[1]+1))
      
      da_ds = self.act_func.backward(self.score)
      
      dloss_ds = np.multiply (dloss_da, da_ds)
      # print(f"dloss_ds: {dloss_ds.shape}")
      
      ds_dw = activated_prev_b
      # print(f"ds_dw: {ds_dw.shape}")
      
      dloss_dw = np.dot (dloss_ds.T, ds_dw)
      # print(f"dloss_dw: {dloss_dw.shape}")
      
      dloss_da_prev = np.dot (dloss_ds,self.weight)
      # print(f"dloss_da_prev: {dloss_da_prev.shape}")
      
      dloss_da_prev = dloss_da_prev[:, :-1]
      # print(f"dloss_da_prev: {dloss_da_prev.shape}")
      
      return dloss_dw, dloss_da_prev
   
   def update(self,dloss_dw, lr, weight_decay=0.):
      # edit
      # print(self.weight.shape)
      
      # print(dloss_dw.shape)
      
      self.weight = (1 - weight_decay) * self.weight - np.multiply(lr, dloss_dw)
   
   def __repr__(self):
      return f"LinearLayer(in_dim={self.in_dim}, out_dim={self.out_dim}, activation={self.act_func})"
