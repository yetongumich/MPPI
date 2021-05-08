from environment import CartPole
from mppi import MPPI
from fgpi import FGPI
import numpy as np

# construct environment
dt = 0.02
env = CartPole(dt=dt)

# construct mppi
num_rollouts = 100
control_horizon = 50
v = 100
rho_sqrtinv=0.5
param_lambda = rho_sqrtinv ** 2 / dt
# param_lambda = 100

total_steps = 200

def run_mppi():
    mppi = MPPI(K=num_rollouts, N=control_horizon, v=v, dt=dt, rho_sqrtinv=rho_sqrtinv, param_lambda=param_lambda)
    curr_state = env.init_state()
    state_seq = [curr_state]
    total_cost = 0
    nominal_controls = np.zeros(control_horizon)
    for k in range(total_steps):
        print(k)
        # compute control 
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

def run_fgpi():
    fgpi = FGPI(N=control_horizon, rho_sqrtinv=rho_sqrtinv, param_lambda=param_lambda, dt=dt)
    curr_state = env.init_state()
    state_seq = [curr_state]
    total_cost = 0
    
    nominal_controls = np.zeros(control_horizon)
    # nominal_controls[:10] += 10
    # nominal_controls[10:20] -=10
    # nominal_controls[20:30] +=15 
    # nominal_controls[30:40] -=15
    # nominal_controls[40:50] +=5
    nominal_states = env.simulate(curr_state, nominal_controls, dt, rho_sqrtinv*np.sqrt(v))

    for k in range(200):
        print(k)
        # compute control
        # rate = 1000 - k * 5
        
        curr_control, nominal_states = fgpi.compute_control(curr_state, nominal_states)
        print("current controls")
        print(curr_control)

        # execute first control
        new_state = env.step(curr_state, curr_control, dt)

        # store step info
        state_seq.append(new_state)
        total_cost += dt*(env.state_cost(new_state) + 0.5 * env.control_cost(curr_control, curr_control))

        # update for next step
        curr_state = new_state
        nominal_states = nominal_states[1:]
        last_state = nominal_states[-1]
        last_next_state = env.step(last_state, 0, dt, 0)
        nominal_states.append(last_next_state)

    print("total cost: ", total_cost)
    env.show_animation(state_seq, dt, step=1)

if __name__ == "__main__":
    run_fgpi()