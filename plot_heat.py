import pandas as pd
import matplotlib.pyplot as plt

errors = pd.read_csv("heat_errors.csv", index_col=0)

reference_orders = [4, 10]

for order in reference_orders:
    errors['time_step^' + str(order)] = [t
                                         ** order for t in errors['time_step']]


y = ['error'] + ['time_step^' + str(order) for order in reference_orders]
plt.close('all')
errors.plot(x='time_step', y=y, logy=True, logx=True, style='-o')


plt.show()
