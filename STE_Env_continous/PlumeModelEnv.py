import gym
from gym import spaces

import numpy as np
import matplotlib.pyplot as plt
from .core import hLikePlume, mcmcPF, resampling_index
from .utils_env import fDyn, gCon

from .core import sensorModel

from .plot_ import drawPlume


class PlumeEnvironment(gym.Env):
    def __init__(self, length, width, height, max_step, startingPosition, render=False):
        super(PlumeEnvironment, self).__init__()

        self.length = length
        self.width = width
        self.height = height
        self.max_step = max_step
        self.startingPosition = startingPosition
        self.source_param = {
            "Q": 5,  # Release rate per time unit
            # source coordinates
            "x": 15.5,
            "y": 16.6,
            "z": 0,
            "u": 2,  # wind speed
            "phi": 45 * np.pi / 180,  # wind direction
            "ci": 2,
            "cii": 10,
        }
        self.sensor_param = {
            "thresh": 5e-10,  # sensor threshold
            "Pd": 0.7,  # probability of detection
            "sig": 1e-4,  # minimum sensor noise
            "sig_pct": 0.5,  # the standard deviation is a percentage of the concentration level
        }
        self.noise = sigma = {
            'x': 0.2,
            'y': 0.2,
            'z': 0.1,
            'Q': 0.2,
            'u': 0.2,
            'phi': 2 * np.pi / 180,
            'ci': 0.1,
            'cii': 0.5
        }
        self.render_if = render
        self.domain = [0, length, 0, width, 0, height]
        self.goal = np.array([self.source_param["x"], self.source_param["y"], self.source_param["z"]])

        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(1,))  # Define the action space
        self.observation_space = spaces.Box(low=np.zeros(3), high=np.array([length, width, height]))

        self.N = 2000  # number of particles
        self.speed = 1
        self.threshold = 0.2
        self.estimated_list = []
        self.action_list = []

        self.count_eps = 0

        # ...

    def reset(self, nsg):
        self.count_eps += 1
        nsg_n = nsg.next_seed()
        np.random.seed(nsg_n)

        self.action_list = []
        self.pos_list = []
        # Reset the environment to an initial state and return the initial observation
        # Initialize any other variables if needed

        obs = self.startingPosition
        self.theta = {
            "x": 0 + self.length * np.random.rand(self.N),
            "y": 0 + self.width * np.random.rand(self.N),
            "z": 0 + 5 * np.random.rand(self.N),
            "Q": np.random.gamma(2, 5, self.N),
            "u": self.source_param["u"] + np.random.randn(self.N) * 2,
            "phi": self.source_param["phi"] * 0.9 + np.random.randn(self.N) * 10 * np.pi / 180,
            "ci": self.source_param["ci"] + 2 * np.random.rand(self.N),
            "cii": self.source_param["cii"] + 2 * np.random.rand(self.N) - 2,
        }
        self.Wpnorm = np.ones(self.N) / self.N

        self.position_history = self.startingPosition.copy()  # for plot
        self.position_curr = self.startingPosition.copy()
        self.pos = {"x_matrix": self.startingPosition[0], "y_matrix": self.startingPosition[1],
                    "z_matrix": self.startingPosition[2]}
        self.D = []

        self.step_count = 0

        if self.render_if:
            plt.cla()

        return obs

    def step(self, action, nsg):

        nsg_n = nsg.next_seed()

        done = False

        action = np.clip(action, -np.pi, np.pi)

        # self.action_list.append(action)

        sensor_data = sensorModel(self.source_param, self.pos, self.sensor_param, nsg)

        # if sensor_data > 0:
        # print("sensor_data:", sensor_data)
        # print("x_matrix", self.position_curr[0], "y_matrix", self.position_curr[1],
        # "z_matrix", self.position_curr[2])

        self.D.append(sensor_data)

        # print("sensor_data:", sensor_data)

        self.theta, self.Wpnorm, _ = mcmcPF(self.theta, self.Wpnorm, sensor_data, fDyn, self.noise, hLikePlume,
                                            self.sensor_param, self.pos, None, nsg, gCon)

        self.position_curr, reward_ = self._next_pos(action)

        self.pos_list.append(self.position_curr)
        self.action_list.append(action)


        # print("self.position_curr:", self.position_curr)

        # update position
        self.pos = {"x_matrix": self.position_curr[0], "y_matrix": self.position_curr[1],
                    "z_matrix": self.position_curr[2]}

        if self.render_if:
            self.render(nsg)

        self.position_history = np.vstack((self.position_history, self.position_curr))

        _, indx = resampling_index(self.Wpnorm, nsg)
        Covar = np.cov(np.array([self.theta['x'][indx], self.theta['y'][indx]]))
        Spread = np.sqrt(np.trace(Covar))

        self.step_count += 1

        if self.step_count >= self.max_step:
            done = True

        info = {}

        if Spread < self.threshold:  # np.array_equal(self.goal, self.position_curr):
            done = True
            reward = 100
            print("done")
            # print("goal! cur_position:", self.position_curr, ',step count:', self.step_count)

            estimated_x = np.mean(self.theta['x'])
            estimated_y = np.mean(self.theta['y'])
            # print("estimated x", estimated_x)
            # print("estimated y", estimated_y)

            self.estimated_list.append((estimated_x, estimated_y))
            # print("action = ", self.action_list)
            # print("action = ", self.action_list)
            # print("pos_list = ", self.pos_list)

            # if self.count_eps >= 6000:
            #     print("action = ", self.action_list)

        # elif sensor_data_ > max(self.D):
        #     reward = 1
        else:
            reward = -1

        reward = reward + reward_

        return self.position_curr, reward, done, info

    def render(self, nsg):

        # nsg_n = nsg.next_seed()

        fig3 = plt.figure()
        ax3 = fig3.add_subplot(111, projection="3d")

        # fig4 = plt.figure()
        # ax4 = fig4.add_subplot(111, projection="3d")

        drawPlume(ax3, self.source_param, self.domain, nsg)
        # drawPlume(ax4, self.source_param, self.domain)

        S = np.zeros_like(self.D)
        for i in range(len(self.D)):
            S[i] = 5  # + np.ceil(self.D[i] * 5e3)

        # print(self.position_history)

        # plot
        ax3.scatter(self.theta['x'], self.theta['y'], self.theta['z'], c='g', marker='o', s=2, alpha=0.03)
        ax3.plot(self.pos['x_matrix'], self.pos['y_matrix'], self.pos['z_matrix'], 'ro', markerfacecolor='r',
                 markersize=5)
        ax3.plot(*self.position_history.T, 'r-')
        ax3.scatter(*self.position_history.T, c='red', marker='o', s=S)
        ax3.scatter(self.source_param['x'], self.source_param['y'], self.source_param['z'], c='black', marker='s', s=20)
        ax3.view_init(90, -90)
        # ax4.view_init(90, -90)
        plt.show()

    def _next_pos(self, action):

        #next_pos = self.position_curr.copy()
        next_pos = np.zeros(3)

        #print("next_pos:", next_pos)

        x = np.cos(action)
        y = np.sin(action)

        # print("x=", x, ",y=", y, ",action:", action)

        next_pos[0] = self.position_curr[0] + x * self.speed
        next_pos[1] = self.position_curr[1] + y * self.speed
        next_pos[2] = 0

        # print("next_pos[0]:", next_pos[0])
        # print("next_pos[1]:", next_pos[1])

        next_pos, reward_ = self._pos_check(next_pos)

        # print("next_pos after:", next_pos)
        #
        # print('\n')

        return next_pos, reward_

    def _pos_check(self, pos):

        reward_ = 0

        if pos[0] < 0 or pos[0] > self.length:
            pos[0] = self.length if pos[0] > self.length else 0
            reward_ = -0
        if pos[1] < 0 or pos[1] > self.width:
            pos[1] = self.width if pos[1] > self.width else 0
            reward_ = -0
        if pos[2] < 0 or pos[2] > self.height:
            pos[2] = self.height if pos[2] > self.height else 0
            reward_ = -0

        return pos, reward_
