import gym
from gym import spaces
import numpy as np
import copy


class RobotArmEnv(gym.Env):
    def __init__(self, number_of_joints=2, max_steps=1000):
        self._number_of_joints = number_of_joints
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, (number_of_joints,))
        self.observation_space = spaces.Box(-1.0, 1.0, (4 + number_of_joints,))

    def _get_state(self):
        return np.concatenate([self._goal, self._angles, self._position], axis=-1)

    def _get_position(self):
        position = np.zeros((2,))
        angle = 0.0
        for joint_idx in range(self._number_of_joints):
            angle += self._angles[joint_idx]
            position += np.asarray([np.sin(angle), np.cos(angle)])
        return position

    def reset(self):
        self._goal = np.random.uniform(-1.0, 1.0, (2,))
        self._angles = np.random.uniform(-np.pi, np.pi, (self._number_of_joints,))
        self._position = self._get_position()
        self._step = 0
        return self._get_state()

    def step(self, action):
        self._step += 1
        terminal = False
        if self._step >= self.max_steps:
            terminal = True
        clipped_action = np.clip(action, -1.0, 1.0)
        angle_change = 0.02 * clipped_action * np.pi
        self._angles = (self._angles + np.pi + angle_change) % (2 * np.pi) - np.pi
        position = self._get_position()
        self.vector_reward = np.abs(self._position - self._goal) - np.abs(position - self._goal)
        reward = np.sum(self.vector_reward)
        self._position = position
        if np.all(np.less(np.abs(self._position - self._goal), 0.1)):
            return self._get_state(), reward, True, {}
        return self._get_state(), reward, terminal, {}


class AngleEnv(gym.Env):
    def __init__(self, dimensionality=4, max_steps=1000):
        self._dimensionality = dimensionality
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, (dimensionality,))
        self.observation_space = spaces.Box(-1.0, 1.0, (dimensionality,))

    def reset(self):
        self._position = np.random.uniform(-1.0, 1.0, (self._dimensionality,))
        self._step = 0
        return self._position

    def step(self, action):
        self._step += 1
        terminal = False
        if self._step >= self.max_steps:
            terminal = True
        clipped_action = np.clip(action, -1.0, 1.0)
        angle = (clipped_action + 1.0) * np.pi / 2.0
        angle_sin = np.sin(angle)
        angle_cos = np.cos(angle)
        actual_action = np.zeros(action.shape)
        for dim in range(0, self._dimensionality, 2):
            r = 0.1 * clipped_action[dim]
            actual_action[dim] = r * angle_sin[dim + 1]
            actual_action[dim + 1] = r * angle_cos[dim + 1]
        position = np.clip(self._position + actual_action, -1.0, 1.0)
        self.vector_reward = np.abs(self._position) - np.abs(position)
        reward = np.sum(self.vector_reward)
        self._position = position
        if np.all(np.less(np.abs(self._position), 0.1)):
            return self._position, reward, True, {}
        return self._position, reward, terminal, {}


class DimChooserEnv(gym.Env):
    def __init__(self, dimensionality=4, max_steps=1000):
        self._dimensionality = dimensionality
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1.0, 1.0, (dimensionality + 1,))
        self.observation_space = spaces.Box(-1.0, 1.0, (dimensionality,))  # 3 *

    def _get_state(self):
        return self._position

    def reset(self):
        self._position = np.random.uniform(-1.0, 1.0, (self._dimensionality,))
        self._step = 0
        return self._get_state()

    def step(self, action):
        self._step += 1
        terminal = False
        if self._step >= self.max_steps:
            terminal = True
        step_size = 0.1 * np.clip(action[0], -1.0, 1.0)
        dim = np.argmax(action[1:])
        position = copy.copy(self._position)
        position[dim] = np.clip(self._position[dim] + step_size, -1.0, 1.0)
        self.vector_reward = np.abs(self._position) - np.abs(position)
        reward = np.sum(self.vector_reward)
        self._position = position
        if np.all(np.less(np.abs(self._position), 0.1)):
            return self._get_state(), reward, True, {}
        return self._get_state(), reward, terminal, {}