<<<<<<< HEAD
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

data[data["precon"] == "multi"][["time_minres", "time_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
data[data["precon"] == "multi"][["time_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
plt.show()

data[data["precon"] == "multi"][["nits_minres", "nits_bpcg"]].unstack(level=0).transpose().plot.bar(subplots=True, figsize=(25, 10), layout=(1, 4), stacked=True)  # .unstack(level=1).plot.bar(subplots=True, figsize=(25, 10),layout=(1,6))
plt.show()

dofs_per_element = data[["ndofs"]].unstack(level=1)
dofs_per_element
=======
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_iterations(data, net_width):
    iterations = data.groupby('net_width').get_group(net_width)\
                     .groupby(['discretization', 'solver']).count()\
                     .unstack().iteration
    axis = iterations.plot.bar(rot=0)
    axis.set_ylabel("iterations")
    axis.set_title(f"iterations by discretization and solver (h={net_width})")


def plot_error_over_iterations(data, net_width, discretization):
    errors = data.groupby(['net_width', 'discretization'])\
                 .get_group((net_width, discretization))\
                 .pivot(index='iteration', columns='solver', values='error')
    axis = errors.plot(logy=True)
    axis.set_ylabel("relative error")
    axis.set_title(f"relative error over iterations (h={net_width}, {discretization})")


def plot_run_time(data, net_width):
    run_time = data.groupby('net_width').get_group(net_width)\
                   .groupby(['discretization', 'solver']).nth(0)\
                   .unstack().run_time
    axis = run_time.plot.bar(rot=0)
    axis.set_ylabel("run time (s)")
    axis.set_title(f"run time (s) by discretization and solver (h={net_width})")


data = pd.read_csv("errors.csv", index_col=0)
plot_iterations(data, net_width=0.01)
plot_run_time(data, net_width=0.01)
plot_error_over_iterations(
    data, net_width=0.01, discretization='HDG BDM 2')
plt.show()
>>>>>>> ee77aba6098f511e49c3fd2fe5943efda41b3fab
