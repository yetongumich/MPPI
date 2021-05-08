""" factor graph path integral implementation """

import gtsam
import gtdynamics as gtd
import numpy as np

class FGPI:
    def __init__(self, N, rho_sqrtinv, param_lambda, dt):
        self.N = N
        prior_sigmas = np.array([1e-5, 1e-5, 1e-5, 1e-5, 1e-5])
        transition_sigmas = np.array([1e-5, 1e-5, 1e-5, 1e-5, 20*rho_sqrtinv/np.sqrt(dt)])
        cost_sigmas = np.sqrt(param_lambda/dt) * np.array([1, 1, 1/np.sqrt(500), 1])
        self.prior_noise = gtsam.noiseModel.Diagonal.Sigmas(prior_sigmas)
        self.transition_noise = gtsam.noiseModel.Diagonal.Sigmas(transition_sigmas)
        self.cost_noise = gtsam.noiseModel.Diagonal.Sigmas(cost_sigmas)
        self.dt = dt

    def state_key(self, i):
        return gtsam.Symbol('x', i).key()

    def compute_control(self, start_state, nominal_states):
        # build graph
        graph = gtsam.NonlinearFactorGraph()
        x0_key = self.state_key(0)
        graph.add(gtd.PriorFactorVector5(x0_key, start_state, self.prior_noise))
        for i in range(self.N):
            x_curr_key = self.state_key(i)
            x_next_key = self.state_key(i+1)
            graph.add(gtd.CPDynamicsFactor(x_curr_key, x_next_key, self.transition_noise, self.dt))
            graph.add(gtd.CPStateCostFactor(x_next_key, self.cost_noise))
        
        # build init values
        init_values = gtsam.Values()
        for i in range(self.N+1):
            x_key = self.state_key(i)
            init_values.insert(x_key, np.array(nominal_states[i]))
        
        # optimize
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(20)
        # params.setVerbosityLM("SUMMARY")
        params.setLinearSolverType("MULTIFRONTAL_QR")
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, init_values, params)
        results = optimizer.optimize()
        # print("error:", graph.error(results))

        # get results
        states = []
        for i in range(self.N + 1):
            x_key = self.state_key(i)
            states.append(results.atVector(x_key))
        
        dx_c = states[1][-1] - states[0][-1]
        u_star = states[0][-1] + 1.1*dx_c/(20*self.dt)
        return u_star, states 