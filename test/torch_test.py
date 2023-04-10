import numpy as np
import torch

a = np.arange(12).reshape(2, 2, 3)
print(a)
b = torch.Tensor(a).permute(2, 1, 0)
print(b)
c = b.permute(2, 1, 0).flip(2)
print(c)
