import numpy as np
from matplotlib import pyplot as plt

plt.axis([-50,50,0,10000])
plt.ion()     # interactive mode on

X = np.arange(-50, 51)
for k in range(1,5):
    print(k)
    Y = [x**k for x in X]
    plt.plot(X, Y)
    plt.draw()
    plt.pause(1)