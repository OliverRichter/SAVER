from model import *


class Agent:
    def __init__(self, obs_shape, action_dimension, hypers):
        self._input_ph = tf.placeholder(tf.float32, [1] + obs_shape, name='observation_input')
        state_size = hypers['state_size']
        self._noise_ph = tf.placeholder(tf.float32, [1, action_dimension], name='noise_input')
        if hypers['env'] == 'dimChooserEnv-v0':
            self._action, self._value, action_mean, action_log_std, _, _ = \
                state_to_dim_select_action(self._input_ph, self._noise_ph, action_dimension, hypers['hidden_layers'])
        else:
            self._action, self._value, action_mean, action_log_std, _, _ = \
                state_to_action(self._input_ph, self._noise_ph, action_dimension, hypers['hidden_layers'])
        self._state_and_action_info = tf.concat([self._input_ph, action_mean, action_log_std], axis=-1)
        if hypers['mode'] == 'saver':
            _, _, self._sc_value = sc_predictor(self._state_and_action_info, hypers['hidden_layers'], state_size)
        else:
            self._sc_value = tf.zeros(obs_shape)
        self._train_op = self._a2c(hypers, obs_shape, action_dimension, state_size)

        self._sess = tf.Session()
        self._sess.run(tf.global_variables_initializer())

    def act(self, obs, noise):
        action, action_info, value, sc_value = self._sess.run([self._action, self._state_and_action_info, self._value,
                                                               self._sc_value],
                                                              feed_dict={self._input_ph: obs, self._noise_ph: noise})
        return action[0], action_info[0], value[0][0], sc_value[0]

    def train(self, obs, noise, actions, advs, Rs, state_changes, sc_advs, sc_Rs):
        self._sess.run(self._train_op, feed_dict={self._batch_input_ph: obs,
                                                  self._batch_noise_ph: noise,
                                                  self._action_ph: actions,
                                                  self._adv_ph: advs,
                                                  self._R_ph: Rs,
                                                  self._state_change_ph: state_changes,
                                                  self._sc_adv_ph: sc_advs,
                                                  self._sc_R_ph: sc_Rs})

    def train_state_change_predictor(self, sampled_action_info, sampled_sc):
        self._sess.run(self._train_sc_predictor, feed_dict={self._state_and_action_info_ph: sampled_action_info,
                                                            self._state_change_ph: sampled_sc})

    def summarize(self, obs, noise, actions, advs, Rs, state_changes, sc_advs, sc_Rs, episode_len, episode_rew, step):
        summary = self._sess.run(self._summary_op, feed_dict={self._batch_input_ph: obs,
                                                              self._batch_noise_ph: noise,
                                                              self._action_ph: actions,
                                                              self._adv_ph: advs,
                                                              self._R_ph: Rs,
                                                              self._state_change_ph: state_changes,
                                                              self._sc_adv_ph: sc_advs,
                                                              self._sc_R_ph: sc_Rs,
                                                              self._episode_length: episode_len,
                                                              self._episode_reward: episode_rew})
        self._summary_writer.add_summary(summary, step)
        self._summary_writer.flush()

    def summarize_pretraining(self, sampled_action_info, sampled_sc, step):
        summary = self._sess.run(self._sc_prediction_loss_summary,
                                 feed_dict={self._state_and_action_info_ph: sampled_action_info,
                                            self._state_change_ph: sampled_sc})
        self._summary_writer.add_summary(summary, step)
        self._summary_writer.flush()

    def _a2c(self, hypers, obs_shape, action_dimension, state_size):
        self._batch_input_ph = tf.placeholder(tf.float32, [hypers['batch_size']] + obs_shape,
                                              name='batch_observation_input')
        self._batch_noise_ph = tf.placeholder(tf.float32, [hypers['batch_size'], action_dimension],
                                              name='batch_noise_input')
        self._action_ph = tf.placeholder(tf.float32, [hypers['batch_size'], action_dimension], name='actions')
        self._adv_ph = tf.placeholder(tf.float32, [hypers['batch_size']], name='advantages')
        self._R_ph = tf.placeholder(tf.float32, [hypers['batch_size']], name='reward_sums')
        self._state_change_ph = tf.placeholder(tf.float32, [hypers['batch_size'], state_size], name='state_changes')
        self._sc_adv_ph = tf.placeholder(tf.float32, [hypers['batch_size'], state_size], name='state_change_advantages')
        self._sc_R_ph = tf.placeholder(tf.float32, [hypers['batch_size'], state_size], name='state_change_reward_sums')

        if hypers['env'] == 'dimChooserEnv-v0':
            _, value, action_mean, action_log_std, neg_log_pi, entropy = \
                state_to_dim_select_action(self._batch_input_ph, self._batch_noise_ph, action_dimension,
                                           hypers['hidden_layers'], reuse=True)
        else:
            _, value, action_mean, action_log_std, neg_log_pi, entropy = \
                state_to_action(self._batch_input_ph, self._batch_noise_ph, action_dimension, hypers['hidden_layers'],
                                reuse=True)
        state_and_action_info = tf.concat([self._batch_input_ph, action_mean, action_log_std], axis=-1)

        if hypers['env'] == 'dimChooserEnv-v0':
            self._state_and_action_info_ph = \
                tf.placeholder_with_default(state_and_action_info,
                                            [hypers['batch_size'], action_dimension + 1 + obs_shape[0]],
                                            name='state_and_action_info')
        else:
            self._state_and_action_info_ph = \
                tf.placeholder_with_default(state_and_action_info,
                                            [hypers['batch_size'], 2 * action_dimension + obs_shape[0]],
                                            name='state_and_action_info')

        policy_loss = tf.reduce_mean(neg_log_pi(self._action_ph) * self._adv_ph)
        entropy_loss = - hypers['entropy_beta'] * tf.reduce_mean(entropy)
        value_loss = hypers['critic_lambda'] * tf.reduce_mean(tf.square(tf.squeeze(value) - self._R_ph))

        a2c_loss = policy_loss + entropy_loss + value_loss
        grads = tf.gradients(a2c_loss, tf.trainable_variables('state_to_action'))
        self._train_sc_predictor = tf.no_op()

        tf.summary.histogram('action_log_std', action_log_std)
        if hypers['env'] == 'dimChooserEnv-v0':
            tf.summary.histogram('step_mean', action_mean[:, 0])
            tf.summary.histogram('step', self._action_ph[:, 0])
            tf.summary.histogram('softmax_output', tf.nn.softmax(action_mean[:, 1:]))
        else:
            tf.summary.histogram('action_mean', action_mean)
            tf.summary.histogram('action', self._action_ph)
        tf.summary.histogram('values', value)
        tf.summary.scalar('policy_loss', policy_loss)
        tf.summary.scalar('value_loss', value_loss)
        tf.summary.scalar('entropy_loss', entropy_loss)
        tf.summary.scalar('base_loss', a2c_loss)

        optimizer = tf.train.RMSPropOptimizer(hypers['learning_rate'])

        # state change prediction and policy loss
        if hypers['mode'] == 'saver':
            stacked_state_and_action_info = tf.concat([self._state_and_action_info_ph] * hypers['K'], axis=0)
            sc_prediction, quantile, sc_value = sc_predictor(stacked_state_and_action_info, hypers['hidden_layers'],
                                                             state_size, reuse=True)
            sc_value = sc_value[:hypers['batch_size']]
            stacked_state_changes = tf.concat([self._state_change_ph] * hypers['K'], axis=0)
            stacked_sc_advs = tf.concat([self._sc_adv_ph] * hypers['K'], axis=0)

            sc_prediction_loss_raw = quantile_loss(quantile, stacked_state_changes - sc_prediction, hypers['huber'])
            sc_prediction_loss = tf.reduce_mean(sc_prediction_loss_raw)
            sc_policy_loss = tf.reduce_mean(sc_prediction_loss_raw * stacked_sc_advs)

            # monotonic loss
            sc_bs_K = tf.transpose(tf.reshape(sc_prediction, [hypers['K'], hypers['batch_size'], state_size]),
                                   [1, 0, 2])
            sc_bs_K_K = tf.stack([sc_bs_K] * hypers['K'], axis=2)
            sc_diff = sc_bs_K_K - tf.transpose(sc_bs_K_K, [0, 2, 1, 3])
            quantile_bs_K = tf.transpose(tf.reshape(quantile, [hypers['K'], hypers['batch_size'], state_size]),
                                         [1, 0, 2])
            quantile_bs_K_K = tf.stack([quantile_bs_K] * hypers['K'], axis=2)
            quantile_diff = quantile_bs_K_K - tf.transpose(quantile_bs_K_K, [0, 2, 1, 3])
            monotonic_loss = tf.where(quantile_diff < 0.0, huber_loss(tf.nn.relu(sc_diff)), tf.zeros(sc_diff.shape))
            tf.summary.scalar('monotonic_loss', tf.reduce_mean(monotonic_loss))
            monotonic_loss = hypers['monotonic_lambda'] * 2.0 * hypers['K'] * tf.reduce_mean(monotonic_loss)

            sc_policy_loss += monotonic_loss

            sc_value_loss = hypers['sc_critic_lambda'] * tf.reduce_mean(tf.square(sc_value - self._sc_R_ph))

            self._sc_prediction_loss_summary = tf.summary.scalar('state_change_prediction_loss', sc_prediction_loss)
            sc_prediction_loss = hypers['prediction_lambda'] * sc_prediction_loss
            tf.summary.scalar('state_change_policy_loss', sc_policy_loss)
            tf.summary.scalar('state_change_value_loss', sc_value_loss)

            state_change_prediction_grads = tf.gradients(sc_prediction_loss, tf.trainable_variables('state_predictor'))
            state_change_value_grads = tf.gradients(sc_value_loss, tf.trainable_variables('state_predictor'))
            policy_grads = tf.gradients(sc_policy_loss, tf.trainable_variables('state_to_action'))
            grads = policy_grads + state_change_value_grads
            self._train_sc_predictor = optimizer.apply_gradients(list(zip(state_change_prediction_grads,
                                                                          tf.trainable_variables('state_predictor'))))

        if hypers['max_gradient_norm'] is not None:
            grads, norm = tf.clip_by_global_norm(grads, hypers['max_gradient_norm'])
            tf.summary.scalar('grad_norm', norm)

        self._episode_length = tf.placeholder(tf.float32, [], name='mean_episode_length')
        self._episode_reward = tf.placeholder(tf.float32, [], name='mean_episode_reward')
        tf.summary.scalar('episode_length', self._episode_length)
        tf.summary.scalar('episode_reward', self._episode_reward)
        self._summary_op = tf.summary.merge_all()

        self._log_dir = hypers['log_dir'] + hypers['run_id']
        self._summary_writer = tf.summary.FileWriter(self._log_dir)
        return optimizer.apply_gradients(list(zip(grads, tf.trainable_variables())))
