import pandas as pd
import matplotlib.pyplot as plt

def plot_over_mesh_size(value, gauss_seidel=True):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('gauss_seidel_enabled')\
        .get_group(gauss_seidel)\
        .pivot(index='mesh_size', columns='order', values=value)\
        .plot(style='-o', logx=True, title=f"{value}, GS={gauss_seidel}")

def plot_over_order(value, gauss_seidel=True):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('gauss_seidel_enabled')\
        .get_group(gauss_seidel)\
        .pivot(index='order', columns='mesh_size', values=value)\
        .plot.bar(title=f"{value}, GS={gauss_seidel}")

def plot_gs_comparison_over_mesh_size(order, value):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('order')\
        .get_group(order)\
        .pivot(index='mesh_size', columns='gauss_seidel_enabled', values=value)\
        .plot(style='-o', logx=True, title=f"{value}, p={order}")

def plot_gs_comparison_over_order(mesh_size, value):
    data = pd.read_csv("data.csv", index_col=0)
    data.groupby('mesh_size')\
        .get_group(mesh_size)\
        .pivot(index='order', columns='gauss_seidel_enabled', values=value)\
        .plot.bar(title=f"{value}, h={mesh_size}")


plt.close('all')
plot_over_mesh_size('time')
plot_over_order('time')
plot_over_mesh_size('iterations')
plot_over_order('iterations')
plot_gs_comparison_over_mesh_size(order=2, value='iterations')
plot_gs_comparison_over_order(mesh_size=1/2, value='iterations')
plot_gs_comparison_over_mesh_size(order=2, value='time')
plot_gs_comparison_over_order(mesh_size=1/2, value='time')
plt.show()
#data = pd.read_csv("templates/data.csv", index_col=0)
#data.pivot(index='mesh_size', columns='order', values='iterations').plot(style='-o', logx=True, ylim=(0, 200))
