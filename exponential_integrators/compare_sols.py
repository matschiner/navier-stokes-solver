import numpy as np
from numpy.linalg import norm
from os import listdir
from os.path import isfile, join
sol = {}
max = 4
for i in range(1, max+1):
    try:
        sol[i] = np.load(f"../sol-wave-{i}.npz")["arr_0"]
    except Exception:
        pass

for i in range(1, max+1):
    print(norm(sol[i] - sol[max]))

