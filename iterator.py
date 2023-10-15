#!/bin/python
import cvxpy as cp
import numpy as np


class Iterator:
    def __init__(self, H):
        """
        H : a matrix, each column h_i is a random channel.
        H.shape : (a_num, s_num).
        a_num : means antennas number.
        s_num : means sensors number.
        init_a : initial solver.
        """
        self.a_num = H.shape[0]
        self.s_num = H.shape[1]
        self.H = cp.Parameter((self.a_num, self.s_num))
        self.H.value = H
        self.init_a = None
        self.c_old = cp.Parameter((2, self.s_num))
        self.c_new = None
        self.cal_init_solver()

        self.init_problem = None
        self.appro_problem = None

    def update_H(self, H):
        self.H.value = H

    def construct_init_problem(self):
        A = cp.Variable((self.a_num, self.a_num), hermitian=True)
        objective = cp.Minimize(cp.trace(A))
        constraints = [A >> 0]

        for k in range(self.s_num):
            hk = self.H[:, k]
            constraints.extend(
                [cp.real(cp.quad_form(hk, A)) >= 1]
            )  # equal to cp.trace(A @ Hk) >= 1
            # Hk = np.outer(hi, np.conj(hk))
            # constraints.extend([cp.real(cp.trace(A @ Hk)) >= 1])

        self.init_problem = cp.Problem(objective, constraints)

    def cal_init_solver(self):
        A = cp.Variable((self.a_num, self.a_num), hermitian=True)

        objective = cp.Minimize(cp.trace(A))
        constraints = [A >> 0]

        for k in range(self.s_num):
            hk = self.H[:, k]
            constraints.extend(
                [cp.real(cp.quad_form(hk, A)) >= 1]
            )  # equal to cp.trace(A @ Hk) >= 1
            # Hk = np.outer(hi, np.conj(hk))
            # constraints.extend([cp.real(cp.trace(A @ Hk)) >= 1])

        problem = cp.Problem(objective, constraints)

        result = problem.solve(solver="SCS", verbose=False)
        print(result)
        A_value = A.value
        if np.linalg.matrix_rank(A_value) == 1:
            self.init_a = None
        else:
            eigenvalues, eigenvectors = np.linalg.eig(A_value)

            eigenvalues_real = eigenvalues.real
            max_index = np.argmax(eigenvalues_real)
            self.init_a = np.sqrt(eigenvalues_real[max_index]) * eigenvectors[max_index]
            print(self.init_a)

        c_old = np.zeros((2, self.s_num))
        for k in range(self.s_num):
            hk = self.H[:, k]
            tmp = np.conj(self.init_a) @ hk
            c_old[:, k] = [tmp.real, tmp.imag]
        self.c_old.value = c_old

    def construct_problem(self):
        pass

    def update(self):
        pass

    def loop(self):
        pass


if __name__ == "__main__":
    seed = 5
    np.random.seed(seed)
    a_num = 3
    s_num = 4
    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    ite = Iterator(H)
