import pandas as pd
import matplotlib.pyplot as plt

def plot_over_mesh_size(value, gauss_seidel=True):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('gauss_seidel_enabled')\
        .get_group(gauss_seidel)\
        .pivot(index='mesh_size', columns='order', values=value)\
        .plot(style='-o', logx=True, title=value)

def plot_over_order(value, gauss_seidel=True):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('gauss_seidel_enabled')\
        .get_group(gauss_seidel)\
        .pivot(index='order', columns='mesh_size', values=value)\
        .plot(style='-o', title=value)


plt.close('all')
plot_over_mesh_size('time')
plot_over_order('time')
plot_over_mesh_size('iterations')
plot_over_order('iterations')
plt.show()
#data = pd.read_csv("templates/data.csv", index_col=0)
#data.pivot(index='mesh_size', columns='order', values='iterations').plot(style='-o', logx=True, ylim=(0, 200))
