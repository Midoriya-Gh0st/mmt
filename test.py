import torch
import torch.nn as nn

x = torch.tensor([1, 2])
y = torch.tensor([3, 4])
print(x, y)
temp = x
x = y
y = temp
print(x, y)
