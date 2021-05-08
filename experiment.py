from environment import CartPole
from mppi import MPPI
from fgpi import FGPI
import numpy as np
import time

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

total_steps = 500

def run_mppi():
    mppi = MPPI(K=num_rollouts, N=control_horizon, v=v, dt=dt, rho_sqrtinv=rho_sqrtinv, param_lambda=param_lambda)
    curr_state = env.init_state()
    state_seq = [curr_state]
    total_cost = 0
    nominal_controls = np.zeros(control_horizon)
    update_times = []
    for k in range(total_steps):
        print(k)
        # compute control
        start_time = time.time()
        new_controls = mppi.compute_controls(env, curr_state, nominal_controls)
        end_time = time.time()
        duration = end_time - start_time
        update_times.append(duration)
        # print("new controls")
        # print(new_controls)
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
    # env.show_animation(state_seq, dt, step=1)
    avg_time = sum(update_times) / len(update_times)
    print("avg time: ", avg_time)
    return total_cost, avg_time

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

    update_times = []

    for k in range(total_steps):
        print(k)
        # compute control
        # rate = 1000 - k * 5
        # nominal_states = env.simulate(curr_state, nominal_controls, dt, rho_sqrtinv*np.sqrt(v))
        start_time = time.time()
        curr_control, nominal_states = fgpi.compute_control(curr_state, nominal_states)
        end_time = time.time()
        duration = end_time - start_time
        update_times.append(duration)
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
    avg_time = sum(update_times) / len(update_times)
    print("avg time: ", avg_time)
    # env.show_animation(state_seq, dt, step=1)
    env.show_trajectory(state_seq, step=1)
    return total_cost, avg_time

if __name__ == "__main__":
    # costs = []
    # avg_times = []
    # for i in range(10):
    #     # cost, avg_time = run_fgpi()
    #     cost, avg_time = run_mppi()
    #     costs.append(cost)
    #     avg_times.append(avg_time)
    # print("average cost: ", sum(costs)/len(costs))
    # print("average time: ", sum(avg_times)/len(avg_times))
    run_fgpi()