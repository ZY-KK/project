import numpy as np


a = np.array([0, 1, 2])
b = np.array([0, 0, -1])
c = []
c.append(a)
c.append(b)
c = np.asarray(c)
# print(np.intersect1d(a,b))
print(len(c))