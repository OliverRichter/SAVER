import numpy as np
import gym
import saver_envs


class Actor:
    def __init__(self, hypers, agent):
        self.env = Monitor(gym.make(hypers['env']))
        self._env_state = self.env.reset()
        self._state_size = hypers['state_size']
        self._act = agent.act
        self._rollout_length = hypers['rollout_length']
        self._discount_factor = hypers['discount']
        self._gae_lambda = hypers['gae_lambda']
        self._no_state_change_predictor = hypers['mode'] == 'a2c'
        self._robot_arm_env = hypers['env'] == 'robotArmEnv-v0'

    def act(self):
        obs = [np.array(self._env_state, copy=False)]
        noise = [np.random.standard_normal(self.env.action_space.shape[0])]
        actions = []
        action_infos = []
        values = []
        rewards = []
        state_changes = []
        state_change_values = []
        state_change_rewards = []
        dones = []
        for step in range(self._rollout_length):
            action, action_info, value, state_change_value = self._act(obs[-1:], noise[-1:])
            self._env_state, reward, done, info = self.env.step(action)
            actions.append(action)
            action_infos.append(action_info)
            values.append(value)
            vector_reward = self.env.unwrapped.vector_reward
            rewards.append(reward)
            if self._no_state_change_predictor:
                state_changes.append(np.zeros(self._state_size))
            else:
                state_changes.append(self._env_state[-self._state_size:] - obs[-1][-self._state_size:])
                state_change_values.append(state_change_value)
                state_change_rewards.append(vector_reward)
            dones.append(done)
            if done:
                self._env_state = self.env.reset()
            obs.append(np.array(self._env_state, copy=False))
            noise.append(np.random.standard_normal(self.env.action_space.shape[0]))
        if not done:
            _, _, value, state_change_value = self._act(obs[-1:], noise[-1:])
            values.append(value)
            state_change_values.append(state_change_value)
        else:
            values.append(0.0)
            state_change_values.append(np.zeros(self._state_size))
        advs = [0.0]
        state_change_advs = [np.zeros(self._state_size)]
        for idx in reversed(range(len(rewards))):
            not_done = 1.0 - float(dones[idx])
            delta = rewards[idx] + self._discount_factor * values[idx + 1] * not_done - values[idx]
            gae = delta + self._discount_factor * self._gae_lambda * not_done * advs[-1]
            advs.append(gae)
            if not self._no_state_change_predictor:
                sc_delta = state_change_rewards[idx] + self._discount_factor * state_change_values[idx + 1] * not_done \
                           - state_change_values[idx]
                sc_gae = sc_delta + self._discount_factor * self._gae_lambda * not_done * state_change_advs[-1]
                state_change_advs.append(sc_gae)
        advs.reverse()
        Rs = np.asarray(advs[:-1]) + np.asarray(values[:-1])
        if self._no_state_change_predictor:
            state_change_advs.extend(state_changes)
            sc_Rs = state_changes
        else:
            state_change_advs.reverse()
            sc_Rs = np.asarray(state_change_advs[:-1]) + np.asarray(state_change_values[:-1])
        return obs[:-1], noise[:-1], actions, action_infos, advs[:-1], Rs, state_changes, state_change_advs[:-1], \
               sc_Rs, info


class Monitor(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._episode_rewards = []
        self._episode_lengths = []
        self._steps = None

    def _reset(self):
        obs = self.env.reset()
        if self._steps is not None:
            self._episode_rewards.append(self._current_reward)
            self._episode_lengths.append(self._steps)
        self._current_reward = 0
        self._steps = 0
        return obs

    def _step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._current_reward += rew
        self._steps += 1
        info['rewards'] = self._episode_rewards
        info['episode_lengths'] = self._episode_lengths
        return obs, rew, done, info