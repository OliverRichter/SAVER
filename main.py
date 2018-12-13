import argparse
import os

from multiprocessing.dummy import Pool as ThreadPool
from utils import *
from actor import Actor
from agent import Agent
import gym
from collections import deque

# side-effect of registering project envs
import saver_envs

p = argparse.ArgumentParser()
p.add_argument(
    '--actors', dest='num_actors', type=int, default=16,
    help='Number of actors')
p.add_argument(
    '--rollout_length', type=int, default=8,
    help='Rollout length')
p.add_argument(
    '--env', default='dimChooserEnv-v0',
    choices=('dimChooserEnv-v0', 'angleEnv-v0', 'robotArmEnv-v0'),
    help='Gym environment')
p.add_argument(
    '--entropy_beta', type=float, default=0.0,
    help='Entroy beta')
p.add_argument(
    '--critic_lambda', type=float, default=0.5,
    help='Critic lambda')
p.add_argument(
    '--sc_critic_lambda', type=float, default=0.5,
    help='SC critic lambda')
p.add_argument(
    '--prediction_lambda', type=float, default=0.1,
    help='Prediction lambda')
p.add_argument(
    '--monotonic_lambda', type=float, default=1.0,
    help='Monotonic lambda')
p.add_argument(
    '--discount', type=float, default=0.99,
    help='Discount')
p.add_argument(
    '--gae_lambda', type=float, default=1.0,
    help='GAE lambda')
p.add_argument(
    '--max_gradient_norm', type=float, default=None,
    help='Max grarient norm')
p.add_argument(
    '--learning_rate', type=float, default=0.01,
    help='Learning rate')
p.add_argument(
    '--mode', default='saver', choices=('saver', 'a2c'),
    help='Training mode')
p.add_argument(
    '--K', type=int, default=32,
    help='K')
p.add_argument(
    '--huber', action='store_true',
    help="Use huber loss")
p.add_argument(
    '--pretrain_batches', dest='number_of_pretrain_batches',
    type=int, default=10000,
    help='Number of pretrain batches')
p.add_argument(
    '--training_batches', dest='number_of_training_batches',
    type=int, default=100000,
    help='Number of training batches')
p.add_argument(
    '--run_id', default=str(np.random.randint(1000)),
    help='Run ID')
p.add_argument(
    '--log_base', default='tensorboard_log',
    help='Base directory for logs')

hypers = dict(p.parse_args()._get_kwargs())
hypers['hidden_layers'] = [256, 128, 64]

print('Run ID: ', str(hypers['run_id']))
hypers['batch_size'] = hypers['rollout_length'] * hypers['num_actors']
assert hypers['mode'] in ['a2c', 'saver']
dummy_env = gym.make(hypers['env'])
print('Action Space: ', dummy_env.action_space)
print(dummy_env.action_space.low)
print(dummy_env.action_space.high)
print('Observation Space: ', dummy_env.observation_space)
print(dummy_env.observation_space.low)
print(dummy_env.observation_space.high)
a_dim = dummy_env.action_space.shape[0]
hypers['state_size'] = dummy_env.observation_space.shape[0]
if hypers['env'] == 'robotArmEnv-v0':
    hypers['state_size'] = 2
hypers['log_dir'] = os.path.join(
    hypers['log_base'],
    hypers['env'],
    'mode={}'.format(hypers['mode']),
    '{}_dim'.format(hypers['state_size']))
print('hypers: ', str(hypers))

agent = Agent(list(dummy_env.observation_space.shape), a_dim, hypers)
actors = []
for idx in range(hypers['num_actors']):
    actors.append(Actor(hypers, agent))
pool = ThreadPool(hypers['num_actors'])


def fill_batch(ai_buffer, sc_buffer):
    state = dummy_env.reset()
    if hypers['env'] == 'dimChooserEnv-v0':
        action_distribution = np.concatenate([np.random.uniform(-1.0, 1.0, size=a_dim),
                                              np.log(np.random.uniform(0.0, 1.0, size=1) + 1e-10)])
        step = action_distribution[:1] + np.exp(action_distribution[-1:]) * np.random.standard_normal(1)
        softmax_dim = softmax(action_distribution[1:-1])
        dim_one_hot = np.zeros(a_dim - 1)
        dim_one_hot[np.random.choice(range(a_dim-1), p=softmax_dim)] = 1.0
        action = np.concatenate([step, dim_one_hot])
    else:
        action_distribution = np.concatenate([np.random.uniform(-1.0, 1.0, size=a_dim),
                                              np.log(np.random.uniform(0.0, 1.0, size=a_dim) + 1e-10)])
        action = action_distribution[:a_dim] + np.exp(action_distribution[a_dim:]) * np.random.standard_normal(a_dim)
    new_state, rew, done, info = dummy_env.step(action)
    ai_buffer.append(np.concatenate([state, action_distribution], axis=-1))
    sc_buffer.append(new_state[-hypers['state_size']:] - state[-hypers['state_size']:])


if hypers['mode'] == 'saver':
    # pretrain state change predictor
    action_info_buffer = deque(maxlen=hypers['batch_size'])
    state_change_buffer = deque(maxlen=hypers['batch_size'])
    for pretrain_step in range(hypers['number_of_pretrain_batches']):
        for _ in range(hypers['batch_size']):
            fill_batch(action_info_buffer, state_change_buffer)
        agent.train_state_change_predictor(action_info_buffer, state_change_buffer)
        if pretrain_step % 50 == 0:
            print('Pretraining with batch ', pretrain_step, ' of ', hypers['number_of_pretrain_batches'])
            agent.summarize_pretraining(action_info_buffer, state_change_buffer, pretrain_step)

step = 0
for training_batch in range(hypers['number_of_training_batches']):
    rollouts = pool.map(lambda actor: actor.act(), actors)
    obs = []
    noise = []
    actions = []
    action_infos = []
    advs = []
    Rs = []
    state_changes = []
    sc_advs = []
    sc_Rs = []
    for rollout in rollouts:
        obs.extend(rollout[0])
        noise.extend(rollout[1])
        actions.extend(rollout[2])
        action_infos.extend(rollout[3])
        advs.extend(rollout[4])
        Rs.extend(rollout[5])
        state_changes.extend(rollout[6])
        sc_advs.extend(rollout[7])
        sc_Rs.extend(rollout[8])
    agent.train(obs, noise, actions, advs, Rs, state_changes, sc_advs, sc_Rs)
    step += hypers['batch_size']
    if training_batch % 50 == 0:
        if rollouts[0][9]['episode_lengths'][-100:]:
            mean_epi_length = np.mean(rollouts[0][9]['episode_lengths'][-100:])
            mean_epi_reward = np.mean(rollouts[0][9]['rewards'][-100:])
        else:
            mean_epi_length = dummy_env.unwrapped.max_steps
            mean_epi_reward = 0.0
        agent.summarize(obs, noise, actions, advs, Rs, state_changes, sc_advs, sc_Rs, mean_epi_length, mean_epi_reward,
                        step)
        print('Run ID: ', str(hypers['run_id']))
        print('hypers: ', str(hypers))
        print('Step: ', step)
        print('Training loop: ', training_batch)
        print('Mean Episode Length: ', mean_epi_length)
        print('Mean Episode Reward: ', mean_epi_reward)

