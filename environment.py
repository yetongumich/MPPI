""" Environment setup. """

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class CartPole:
    def __init__(self):
        """ Set parameters. """
        self.g = 9.81
        self.m_c = 1
        self.m_p=0.01
        self.l=0.25

    def time_derivative(self, state, f_des):
        x = state[0]
        xd = state[1]
        theta = state[2]
        thetad = state[3]
        f = state[4]

        m_all = self.m_c+ self.m_p*np.sin(theta)**2
        f_all = f + self.m_p*np.sin(theta) * (self.l * thetad**2 + self.g*np.cos(theta))
        xdd = f_all / m_all

        moment_all = m_all * self.l
        f1 = -f*np.cos(theta)
        f2 = -self.m_p * self.l * thetad**2*np.cos(theta)*np.sin(theta)
        f3 = -(self.m_c+self.m_p)*self.g*np.sin(theta)
        thetadd = (f1+f2+f3)/moment_all

        fd = 20 * (f_des-f)

        return np.array([xd, xdd, thetad, thetadd, fd])

    
    def step(self, state, f_des, dt):
        state_d = self.time_derivative(state, f_des)
        new_state = state + state_d * dt
        return new_state

    def init_state(self):
        return np.array([0, 0, 0, 0, 0])
    
    def goal_state(self):
        return np.array([0, 0, np.pi, 0, 0])
    
    def show_frame(self, ax, state, i):
        ax.clear()
        x = state[0]
        theta = state[2]
        cart_x = [x-0.1, x+0.1]
        cart_y = [0, 0]
        ax.plot(cart_x, cart_y, color='r')

        pole_x = [x, x+np.sin(theta)*self.l] 
        pole_y = [0, -np.cos(theta) * self.l]
        ax.plot(pole_x, pole_y, color='b')

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-2, 2)
        ax.set_ylim(-1, 1)
        ax.set_title("%.3f"%i)

    def show_animation(self, state_seq, dt, step=1):
        fig = plt.figure(figsize=(20, 10), dpi=80)
        ax = fig.add_subplot(1, 1, 1)

        def animate(i):
            self.show_frame(ax, state_seq[i], dt * i)

        num_steps = len(state_seq)
        frames = np.arange(0, num_steps, step)
        ani = FuncAnimation(fig, animate, frames=frames, interval=10)
        plt.show()
    

if __name__ == "__main__":
    env = CartPole()
    state = env.init_state()
    state_seq = [state]
    num_steps = 2000
    dt = 0.001
    f_des_seq = np.zeros(num_steps)
    f_des_seq[:100] += 10
    for i in range(num_steps):
        new_state = env.step(state, f_des_seq[i], dt)
        state = new_state
        state_seq.append(state)
    # print(state_seq)
    env.show_animation(state_seq, dt, step=10)
