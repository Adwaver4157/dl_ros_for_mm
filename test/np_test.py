import numpy as np

a = np.arange(12).reshape(2, 2, 3)

print(a)

print(a[..., ::-1])
