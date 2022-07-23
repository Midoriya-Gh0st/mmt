import torch
import torch.nn as nn
import numpy as np

# 检查 weight - head
"""
weight = nn.Parameter(torch.ones(3), requires_grad=False)
print("[w1]", weight)

weight[0] = 0.5
print("[w2]:", weight)

heads_list = torch.ones(2, 3, 4)  # [bsz, head, dim]
print("[h1]:", heads_list)

weight1 = weight.reshape(1, -1, 1)
print("[w3]:", weight1, weight1.shape)

weight2 = weight1
    # .repeat([2, 1, 1])
print("[w4]:", weight2, weight2.shape)

print(heads_list.shape, weight2.shape)
out = heads_list * weight2
# out = torch.matmul(heads_list, weight2.transpose(1, 2))
print("[h2]:", out, out.shape)
"""

# 检查 permute:
n1 = torch.tensor([[1, 1, 1], [2, 2, 2]])
print(n1.shape, n1)

n2 = n1.permute(1, 0)
print(n2.shape, n2)
