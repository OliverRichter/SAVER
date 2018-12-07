from utils import *


def state_to_action(inpt, noise, action_dimension, hidden_layers, name='state_to_action', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        hidden = tf.layers.flatten(inpt)
        for hidden_dim in hidden_layers:
            hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
        pdparams_mean = tf.layers.dense(hidden, action_dimension, activation=None)
        # pdparams_log_std = tf.layers.dense(hidden, action_dimension, activation=None)
        pdparams_log_std = tf.get_variable(name="logstd", shape=[1, action_dimension],
                                           initializer=tf.constant_initializer(-1.0))
        std = tf.clip_by_value(tf.exp(pdparams_log_std), 1e-20, 1e20)
        neg_action_log_likelihood = lambda action_taken: gauss_neg_log_pi(action_taken, pdparams_mean, std,
                                                                          pdparams_log_std)
        entropy = tf.reduce_sum(pdparams_log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)
        action = pdparams_mean + noise * std
        value = tf.layers.dense(hidden, 1, activation=None)
        return action, value, pdparams_mean, pdparams_mean * 0.0 + pdparams_log_std, neg_action_log_likelihood, entropy


def state_to_dim_select_action(inpt, noise, action_dimension, hidden_layers, name='state_to_action', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        hidden = tf.layers.flatten(inpt)
        for hidden_dim in hidden_layers:
            hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
        action_mean_step_and_dim_logits = tf.layers.dense(hidden, action_dimension, activation=None)
        step_mean = action_mean_step_and_dim_logits[:, :1]
        dim_logits = action_mean_step_and_dim_logits[:, 1:]
        step_log_std = tf.get_variable(name="logstd", shape=[1, 1], initializer=tf.constant_initializer(-1.0))
        std = tf.clip_by_value(tf.exp(step_log_std), 1e-20, 1e20)

        def neg_action_log_likelihood(action_taken):
            gaussian_neg_log_pi = gauss_neg_log_pi(action_taken[:, :1], step_mean, std, step_log_std)
            categorical_neg_log_pi = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=action_taken[:, 1:],
                                                                                           logits=dim_logits), axis=-1)
            return tf.expand_dims(gaussian_neg_log_pi + categorical_neg_log_pi, axis=-1)

        softmax_dim = tf.nn.softmax(dim_logits)
        entropy = -tf.reduce_sum(tf.log(softmax_dim + 1e-10) * softmax_dim, axis=-1)
        dim_one_hot = tf.reshape(tf.one_hot(tf.multinomial(dim_logits, 1), action_dimension -1),
                                 [-1, action_dimension - 1])
        action = tf.concat([step_mean + noise[:, :1] * std, dim_one_hot], axis=-1)
        value = tf.layers.dense(hidden, 1, activation=None)
        pdparams_mean = tf.concat([step_mean, dim_logits], axis=-1)
        return action, value, pdparams_mean, step_mean * 0.0 + step_log_std, neg_action_log_likelihood, entropy


def sc_predictor(state_and_action_input, hidden_layers, state_size, name='state_predictor', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        policy_hidden = state_and_action_input
        policy_hidden = tf.layers.dense(policy_hidden, units=hidden_layers[0], activation=tf.nn.relu)
        full_hidden = tf.split(tf.layers.dense(policy_hidden, units=state_size*hidden_layers[1], activation=tf.nn.relu),
                               state_size, axis=-1)

        quantile = tf.random_uniform([int(state_and_action_input.shape[0]), state_size])
        quantile_expanded = tf.split(expand_dim_cos(quantile, state_size * hidden_layers[1]), state_size, axis=-1)
        state_change_prediction = []
        for hidden, quant in zip(full_hidden, quantile_expanded):
            policy_hidden = hidden * quant
            for hidden_dimension in hidden_layers[2:]:
                policy_hidden = tf.layers.dense(policy_hidden, units=hidden_dimension, activation=tf.nn.relu)
            state_change_prediction.append(tf.layers.dense(policy_hidden, 1, activation=None))
        state_change_prediction = tf.concat(state_change_prediction, axis=-1)

        hidden = state_and_action_input
        for hidden_dim in hidden_layers:
            hidden = tf.layers.dense(hidden, hidden_dim, activation=tf.nn.relu)
        value = tf.layers.dense(hidden, state_size, activation=None)
        return state_change_prediction, quantile, value