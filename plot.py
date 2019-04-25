import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame()
wrong_precons = pd.read_csv("bpcg_test_precons.csv")
nc = pd.read_csv("bpcg_test_all_nc.csv")
hdg = pd.read_csv("bpcg_test_all_hdg.csv")
hdg["precon"] = "bddc"
nc["precon"] = "bddc"

data = data.append(hdg)
data = data.append(nc)

wrong_precons.iloc[1::2, wrong_precons.columns.get_loc("precon")] = "multi"

data = data.append(wrong_precons)
data.set_index(["method", "vert"], inplace=True)

data[["ndofs", "nits_bpcg", "nits_minres"]] = data[["ndofs", "nits_bpcg", "nits_minres"]].astype(int)
p = data.unstack(level=0)
p.plot()
data[["nits_minres", "nits_bpcg"]].unstack(level=0).plot()
data[["nits_bpcg"]].unstack(level=0).plot()
plt.show()

data[["time_bpcg"]].unstack(level=0).plot()
data[["time_minres", "time_bpcg"]].unstack(level=0).plot()

data[data["precon"] == "bddc"][["time_minres", "time_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
data[data["precon"] == "multi"][["time_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
plt.show()

data[data["precon"] == "multi"][["nits_minres", "nits_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
plt.show()

dofs_per_element = data[["ndofs"]].unstack(level=1)
dofs_per_element
