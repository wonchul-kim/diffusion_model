import torch 

data = torch.Tensor([1, 2, 3])
print(data)

_data = torch.rand_like(data)
print(_data)