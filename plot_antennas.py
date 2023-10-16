#!/bin/python

import matplotlib.pyplot as plt
import numpy as np

antennas = np.load("antennas.npy")

x = range(1, 11)

plt.plot(x, 10 * antennas.real - 20, marker="o")
plt.xlabel("The number of FC antennas.")
plt.ylabel("MSE(dB)")
plt.xticks(range(1, 11))
plt.yticks(range(-23, -1, 2))
plt.xlim(0.5, 10.5)
plt.ylim(-23, -3)
plt.grid(True)
plt.show()
