""" MPPI implementation. """

import numpy as np

class MPPI:
    def __init__(self, K, N, v, dt, rho_sqrtinv, param_lambda):
        """
        Args:
            K (int): number of samples
            N (int): number of time steps
            v (float): exploration variance
        """
        self.K = K
        self.N = N
        self.v_coeff = v
        self.dt = dt
        self.rho_sqrtinv = rho_sqrtinv
        self.param_lambda = param_lambda

    def generate_control_variations(self):
        sigma = np.sqrt(self.v_coeff) * self.rho_sqrtinv
        delta_u_mat = np.random.normal(0, sigma, size=(self.N, self.K))
        return delta_u_mat

    def run_trajectory(self, env, start_state, controls):
        x = start_state
        x_seq = [x]
        for i in range(self.N):
            u = controls[i]
            x_next = env.step(x, u, self.dt, self.rho_sqrtinv)
            x = x_next
            x_seq.append(x)
        return x_seq

    def compute_q_tilde(self, env, state, u, delta_u):
        q = env.state_cost(state)
        term1 = (1 - 1./self.v_coeff)/2 * env.control_cost(delta_u, delta_u)
        term2 = env.control_cost(u, delta_u)
        term3 = 0.5 * env.control_cost(u, u)
        q_tilde = q + term1 + term2 + term3
        return q_tilde
        

    def compute_trajectory_S(self, env, x_seq, controls, delta_us):
        next_x_seq = x_seq[1:]
        curr_x_seq = x_seq[:-1]
        S = 0
        S_seq = []
        for i in range(len(controls)):
            S += self.compute_q_tilde(env, next_x_seq[i], controls[i], delta_us[i])
            S_seq.append(S)
        # print("cost: ", S)
        S_seq = [S] * len(controls)
        return S_seq
    
    def update_controls(self, nominal_controls, delta_u_mat, S_mat):
        new_controls = np.array(nominal_controls)
        for i in range(len(nominal_controls)):
            S_i = S_mat[i,:]
            S_i -= np.min(S_i)
            weights = np.exp(-1.0/self.param_lambda * S_mat[i,:])
            delta_u = np.sum(delta_u_mat[i,:] * weights) / np.sum(weights)
            new_controls[i] += delta_u
        return new_controls

    def compute_controls(self, env, start_state, nominal_controls):

        delta_u_mat = self.generate_control_variations()
        S_mat = []
        for k in range(self.K):
            delta_us = delta_u_mat[:, k]
            controls = nominal_controls + delta_us
            x_seq = self.run_trajectory(env, start_state, controls)
            S_seq = self.compute_trajectory_S(env, x_seq, controls, delta_us)
            S_mat.append(S_seq)
        S_mat = np.array(S_mat).transpose()
    
        new_controls = self.update_controls(nominal_controls, delta_u_mat, S_mat)

        return new_controls


