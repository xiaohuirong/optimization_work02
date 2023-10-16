#!/bin/python
import cvxpy as cp
import numpy as np


class Iterator:
    def __init__(self, a_num, s_num):
        """
        H : a matrix, each column h_i is a random channel.
        H.shape : (a_num, s_num).
        a_num : means antennas number.
        s_num : means sensors number.
        init_a : initial solver.
        """
        self.eps = 1e-5
        self.max_ite_time = 20
        self.a_num = a_num
        self.s_num = s_num
        self.H = None

        self.H_real = None
        self.H_imag = None
        self.A = None
        self.init_problem = None

        self.a = None
        self.C_OLD = None
        self.problem = None
        self.result = None
        self.init_result = None

        self.construct_init_problem()
        self.construct_problem()

    def load_channel(self, H):
        if self.a_num != H.shape[0] or self.s_num != H.shape[1]:
            print("H' shape don't match antennas X sensors")
            return 1

        else:
            self.H = H
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
            constraints.extend(
                [cp.real(cp.quad_form(hk, self.A)) >= 1]
            )  # equal to cp.trace(A @ Hk) >= 1

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
        self.H_real.value = self.H.real
        self.H_imag.value = self.H.imag

        self.init_result = self.init_problem.solve(solver="SCS", verbose=False)
        A_value = self.A.value
        eigenvalues, eigenvectors = np.linalg.eig(A_value)
        # try:
        #    eigenvalues, eigenvectors = np.linalg.eig(A_value)
        # except:
        #    print(self.init_result)
        #    print(self.A.value)
        #    print(self.H)
        eigenvalues_real = eigenvalues.real
        max_index = np.argmax(eigenvalues_real)
        self.init_a = np.sqrt(eigenvalues_real[max_index]) * eigenvectors[max_index]

        if np.linalg.matrix_rank(A_value) == 1:
            return 0

        C_OLD = np.zeros((2, self.s_num))
        for k in range(self.s_num):
            hk = self.H[:, k]
            tmp = np.conj(self.init_a) @ hk
            C_OLD[:, k] = [tmp.real, tmp.imag]
        self.C_OLD.value = C_OLD
        return 1

    def cal_solver(self):
        self.result = self.problem.solve(solver="SCS", verbose=False)
        C_NEW = np.zeros((2, self.s_num))
        for k in range(self.s_num):
            hk = self.H[:, k]
            tmp = np.conj(self.a.value) @ hk
            C_NEW[:, k] = [tmp.real, tmp.imag]
        self.C_NEW = C_NEW

    def loop(self):
        if ~self.cal_init_solver():
            self.a.value = self.init_a
            self.result = self.init_result
            return
        count = 0
        while True:
            self.cal_solver()
            count += 1
            diff = np.linalg.norm(self.C_NEW - self.C_OLD.value)
            if diff <= self.eps or count >= self.max_ite_time:
                break
            self.C_OLD.value = self.C_NEW


if __name__ == "__main__":
    # seed = 1
    # np.random.seed(seed)
    a_num = 1
    s_num = 15
    H = (np.random.randn(a_num, s_num) + 1j * np.random.randn(a_num, s_num)) / np.sqrt(
        2
    )
    H = np.array(
        [
            [
                -1.51473207 + 0.07613419j,
                -0.27382313 + 0.40333406j,
                0.47853167 + 0.15210837j,
                -0.64631679 + 1.08877744j,
                -0.09271474 - 0.49672768j,
                -0.87575654 + 1.04165954j,
                -0.06661203 + 0.13203127j,
                -1.08685733 - 0.30858174j,
                -0.9051136 + 0.7821169j,
                2.29892388 + 0.79503577j,
                0.10647928 + 0.13058578j,
                0.53828483 - 0.82748696j,
                -0.00365755 - 0.31346699j,
                -0.58732198 + 1.95502388j,
                -1.13413035 - 0.87685706j,
            ]
        ]
    )
    ite = Iterator(H.shape[0], H.shape[1])
    ite.load_channel(H)
    ite.loop()
    print(ite.result)
    print(ite.a.value)
