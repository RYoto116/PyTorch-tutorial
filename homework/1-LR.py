import numpy as np
import matplotlib.pyplot as plt

x_data = [1., 2., 3.]
y_data = [3., 5., 7.]
n_sample = len(x_data)

w_list = np.arange(0.0, 4.0, 0.1)
b_list = np.arange(0.0, 2.0, 0.1)

ww, bb = np.meshgrid(w_list, b_list)

loss_val = np.zeros_like(ww)
for x_val, y_val in zip(x_data, y_data):
    z_val = ww * x_val + bb
    loss_val += (z_val - y_val) * (z_val - y_val) / n_sample

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(ww, bb, loss_val)
plt.show()