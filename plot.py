import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv("bpcg_test.csv")
data.set_index(["method","vert"],inplace=True)
p=data.unstack(level=0)
p.plot()
data[["nits_minres","nits_bpcg"]].unstack(level=0).plot()
data[["nits_bpcg"]].unstack(level=0).plot()
plt.show()

data[["time_bpcg"]].unstack(level=0).plot()
data[["time_minres","time_bpcg"]].unstack(level=0).plot()
plt.show()


dofs_per_element=data[["ndofs"]].unstack(level=1)
dofs_per_element