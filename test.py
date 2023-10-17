import numpy as np

from nets.activations import ReLU 

relu = ReLU()
print(relu)

relu(relu.forward(np.random.randn(3, 4)))