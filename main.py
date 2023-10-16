#!/bin/python

import multiprocessing

import matplotlib.pyplot as plt
import numpy as np

from iterator import Iterator


def one_simulation(_):
    global a_num, s_num
    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    ite = Iterator(a_num, s_num)
    ite.load_channel(H)
    ite.loop()
    return ite.result


seed = 1
np.random.seed(seed)
a_num = 2
s_num = 3
num_simu = 10000
num_cores = 12
sum_result = 0

with multiprocessing.Pool(processes=num_cores) as pool:
    results = pool.map(one_simulation, range(num_simu))

sum_result = sum(results)
sum_result /= num_simu

print(sum_result)
