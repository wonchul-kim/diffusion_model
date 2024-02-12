import torch
import torch.nn as nn

input_tensor = torch.tensor( 
    [ 
        [1, 1, 2, 4], 
        [5, 6, 7, 8], 
        [3, 2, 1, 0], 
        [1, 2, 3, 4] 
    ], dtype = torch.float32) 

input_tensor = input_tensor.reshape(1, 1, 4, 4)
pool = nn.MaxPool2d(kernel_size=2, stride=2)
output = pool(input_tensor)

print(output)