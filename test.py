import matplotlib.pyplot as plt
import numpy as np
XX=np.arange(-5,5,0.5)
YY=np.arange(-5,5,0.5)
X,Y=np.meshgrid(XX,YY)
Z=np.sin(X)+np.cos(Y)

fig=plt.figure()
ax3=plt.axes(projection='3d')
ax3.plot_surface(X,Y,Z,cmap='rainbow')
ax3.contour(X,Y,Z,zdim='z',offset=-2,cmap='rainbow') #等高线图，要设置offset，为Z的最小值
plt.show()