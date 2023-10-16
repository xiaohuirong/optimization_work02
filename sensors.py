#!/bin/python

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

from iterator import Iterator

seed = 1
np.random.seed(seed)


def one_simulation(_):
    global a_num, s_num
    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    ite = Iterator(a_num, s_num)
    ite.load_channel(H)
    ite.loop()
    return ite.result


a_num = 4
num_simu = 10000
num_cores = 12
sum_results = []
for s_num in range(10, 110, 10):
    print("-------{}--------".format(s_num))
    sum_result = 0

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(one_simulation, range(num_simu))

    sum_result = sum(np.log10(results))
    sum_result /= num_simu
    sum_results.append(sum_result)
    print(sum_result)
    np.save("sensors.npy", sum_results)

print(sum_results)
