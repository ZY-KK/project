import numpy as np



# h={}
# h['a']={'c':1}

# h['a']={'b':2}

# print('b' in h['a'].keys())
b=0

a  = np.array([0.9])
np.clip(a, a_min=0.5, a_max=0.8, out=a)
print(a)