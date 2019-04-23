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
    axis.set_ylabel("error")
    axis.set_title(f"error over iterations (h={net_width}, {discretization})")


data = pd.read_csv("errors.csv", index_col=0)
plot_iterations(data, net_width=0.01)
plot_error_over_iterations(
    data, net_width=0.01, discretization='taylor hood 3')
plt.show()
