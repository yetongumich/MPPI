from environment import CartPole
from mppi import MPPI
import numpy as np

# construct environment
dt = 0.02
env = CartPole(dt=dt)

# construct mppi
num_rollouts = 100
control_horizon = 50
nominal_controls = np.zeros(50)
v = 100
rho_sqrtinv=0.5
param_lambda = rho_sqrtinv ** 2 / dt
mppi = MPPI(K=100, N=control_horizon, v=v, dt=dt, rho_sqrtinv=rho_sqrtinv, param_lambda=param_lambda)


total_steps = 200
curr_state = env.init_state()
state_seq = [curr_state]
total_cost = 0
for k in range(total_steps):
    print(k)
    # compute control 
    # print("nominal controls:")
    # print(nominal_controls)
    new_controls = mppi.compute_controls(env, curr_state, nominal_controls)
    print("new controls")
    print(new_controls)
    # execute first control
    curr_control = new_controls[0]
    new_state = env.step(curr_state, curr_control, dt)
    # store step info
    state_seq.append(new_state)
    total_cost += dt*(env.state_cost(new_state) + 0.5 * env.control_cost(curr_control, curr_control))
    # update for next step
    curr_state = new_state
    nominal_controls = np.append(np.array(new_controls[1:]),0)

    

print("total cost: ", total_cost)
env.show_animation(state_seq, dt, step=1)