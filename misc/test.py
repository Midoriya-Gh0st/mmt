import pickle

import torch
import torch.nn as nn
import numpy as np

# 检查 weight - head

import csv

# 检查id是否相同.
def load_id(f):
    file = f'data/ids/ids_{f}.pkl'
    return list(pickle.load(open(file, 'rb')))
id_train = load_id('train')
id_test = load_id('test')
id_valid = load_id('valid')
id2 = load_id('2')[1:]

id1 = []
id1.extend(id_train)
id1.extend(id_valid)
id1.extend(id_test)
print(id1)
print(id2)
print(id1 == id2)

# ids = {'train': id_train, 'valid': id_valid, 'test': id_test}
# print(ids)



# 导出ids.
"""
file = "data/MOSI-label.csv"
csv_read = csv.reader(open(file))
ids = []
for info in csv_read:
    n1 = info[0]
    n2 = info[1]
    n = f'{n1}_{n2}'
    ids.append(n)
pickle.dump(ids, open(f'ids_2.pkl', 'wb'))
# print(csv_read)
"""

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
"""
n1 = torch.tensor([[1, 1, 1], [2, 2, 2]])
print(n1.shape, n1)

n2 = n1.permute(1, 0)
print(n2.shape, n2)
"""

# 检查 matrix*系数
"""
n1 = torch.ones([2, 3, 4])
print(n1)
n1 = n1 * 0.6
print(n1)
"""
