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
        self.H = H
        self.init_a = None
        self.c_new = None

        self.construct_init_problem()
        self.construct_problem()

    def update_channel(self, H):
        self.H_real.value = H.real
        self.H_imag.value = H.imag

    def construct_init_problem(self):
        self.H_real = cp.Parameter((self.a_num, self.s_num))
        self.H_imag = cp.Parameter((self.a_num, self.s_num))
        self.A = cp.Variable((self.a_num, self.a_num), hermitian=True)
        objective = cp.Minimize(cp.trace(self.A))
        constraints = [self.A >> 0]

        for k in range(self.s_num):
            hk = self.H_real[:, k] + 1j * self.H_imag[:, k]
            # hk_real = self.H_real[:, k]
            # hk_imag = self.H_imag[:, k]
            constraints.extend(
                [cp.real(cp.quad_form(hk, self.A)) >= 1]
                # [
                #    cp.quad_form(hk_real, cp.real(self.A))
                #    + cp.quad_form(hk_imag, cp.real(self.A))
                #    + 2 * hk_imag @ cp.imag(self.A) @ hk_real
                #    >= 1
                # ]
            )  # equal to cp.trace(A @ Hk) >= 1
            # Hk = np.outer(hi, np.conj(hk))
            # constraints.extend([cp.real(cp.trace(A @ Hk)) >= 1])

        self.init_problem = cp.Problem(objective, constraints)

    def construct_problem(self):
        self.C_OLD = cp.Parameter((2, self.s_num))
        self.a = cp.Variable(self.a_num, complex=True)
        objective = cp.Minimize(cp.sum_squares(self.a))
        constraints = []

        for k in range(self.s_num):
            hk = self.H_real[:, k] + 1j * self.H_imag[:, k]
            tmp = cp.conj(self.a) @ hk
            ck = [cp.real(tmp), cp.imag(tmp)]
            ck_old = self.C_OLD[:, k]
            p2 = ck_old[0] * (ck[0] - ck_old[0]) + ck_old[1] * (ck[1] - ck_old[1])

            constraints.extend([cp.sum_squares(ck_old) + 2 * p2 >= 1])
        self.problem = cp.Problem(objective, constraints)

    def cal_init_solver(self):
        self.H_real.value = H.real
        self.H_imag.value = H.imag

        result = self.init_problem.solve(solver="SCS", verbose=False)
        print(result)
        A_value = self.A.value
        if np.linalg.matrix_rank(A_value) == 1:
            self.init_a = None
        else:
            eigenvalues, eigenvectors = np.linalg.eig(A_value)

            eigenvalues_real = eigenvalues.real
            max_index = np.argmax(eigenvalues_real)
            self.init_a = np.sqrt(eigenvalues_real[max_index]) * eigenvectors[max_index]
            print(self.init_a)

        C_OLD = np.zeros((2, self.s_num))
        for k in range(self.s_num):
            hk = self.H[:, k]
            tmp = np.conj(self.init_a) @ hk
            C_OLD[:, k] = [tmp.real, tmp.imag]
        self.C_OLD.value = C_OLD

    def cal_problem(self):
        result = self.problem.solve(solver="SCS", verbose=True)
        print(result)

    def loop(self):
        pass


if __name__ == "__main__":
    seed = 1
    np.random.seed(seed)
    a_num = 3
    s_num = 4
    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    ite = Iterator(H)
    ite.cal_init_solver()
    ite.cal_problem()

    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    ite.update_channel(H)
    ite.cal_init_solver()
