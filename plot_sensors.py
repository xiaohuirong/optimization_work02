#!/bin/python

import matplotlib.pyplot as plt
import numpy as np

sensors = np.load("sensors.npy")

x = range(10, 110, 10)

plt.plot(x, 10 * sensors.real - 20, marker="o")
plt.xlabel("The number of sensors")
plt.ylabel("MSE(dB)")
plt.xticks(range(10, 110, 10))
plt.yticks(range(-20, -14))
#plt.xlim(0.5, 10.5)
#plt.ylim(-23, -3)
plt.grid(True)
plt.show()
