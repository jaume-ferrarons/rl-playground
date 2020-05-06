from tensorflow.compat.v1.keras import models, layers, backend, optimizers
import tensorflow.compat.v1 as tf
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


print("Tensorflow version:", tf.__version__)

tf.reset_default_graph()
sess = tf.InteractiveSession()


class DDPGAgent():
    def __init__(self, obs_size, action_size,
                 actor_model=None,
                 critic_model=None,
                 use_target_network=False,
                 learning_rate=1e-3, reward_discount=0.99, tau=0.001):
        self.obs_size = obs_size
        self.action_size = action_size
        self.use_target_network = use_target_network
        self.lr = learning_rate
        self.rd = reward_discount
        self.tau = tau

        # Create models if not provided
        if actor_model is None:
            actor_model = models.Sequential()
            actor_model.add(layers.Dense(
                16, input_shape=obs_size, activation='relu'))
            actor_model.add(layers.Dense(16, activation='relu'))
            actor_model.add(layers.Dense(16, activation='relu'))
            actor_model.add(layers.Dense(
                action_size, name='action', activation='tanh'))
            actor_model.summary()
        self.actor_model = actor_model

        if critic_model is None:
            state_input = layers.Input(shape=obs_size)
            action_input = layers.Input(shape=action_size)
            all_input = layers.Concatenate()([state_input, action_input])
            h1 = layers.Dense(32, activation='relu')(all_input)
            h2 = layers.Dense(32, activation='relu')(h1)
            h3 = layers.Dense(32, activation='relu')(h2)
            output = layers.Dense(1, name='q-value')(h3)
            critic_model = models.Model(
                inputs=[state_input, action_input], outputs=output)
            critic_model.summary()
        self.critic_model = critic_model

        if use_target_network:
            self.target_network_critic = tf.keras.models.clone_model(
                self.critic_model)
            self.target_network_actor = tf.keras.models.clone_model(
                self.actor_model)

        self.state_ph = tf.placeholder(tf.float32, shape=(None, *obs_size))
        self.actions_ph = tf.placeholder(tf.float32, shape=(None, action_size))
        self.rewards_ph = tf.placeholder(tf.float32, shape=(None))
        self.next_states_ph = tf.placeholder(
            tf.float32, shape=(None, *obs_size))
        self.is_done_ph = tf.placeholder(tf.float32, shape=(None))

        self.loss = self.q_loss(self.state_ph, self.actions_ph,
                                self.rewards_ph, self.next_states_ph, self.is_done_ph)
        aer = self.action_expected_reward(self.state_ph)

        self.train_critic_step = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss,
            var_list=self.critic_model.trainable_variables
        )
        self.train_actor_step = tf.train.AdamOptimizer(learning_rate/10).minimize(
            -aer,
            var_list=self.actor_model.trainable_variables
        )
        sess.run(tf.global_variables_initializer())

    def q_loss(self, states_ph, actions_ph, rewards_ph, next_states_ph, is_done_ph):
        q_values = self.critic_model([states_ph, actions_ph])
        if self.use_target_network:
            next_state_actions = self.target_network_actor(next_states_ph)
            next_state_q_values = self.target_network_critic(
                [next_states_ph, next_state_actions])
        else:
            next_state_actions = self.actor_model(next_states_ph)
            next_state_q_values = self.critic_model(
                [next_states_ph, next_state_actions])
        is_not_done = 1 - is_done_ph
        target_q_values_for_actions = rewards_ph + \
            self.rd*next_state_q_values*is_not_done
        loss = (q_values - tf.stop_gradient(target_q_values_for_actions))**2
        return tf.reduce_mean(loss)

    def get_qa(self, state, action):
        return self.critic_model.predict([[state], [action]])[0][0]

    def get_action(self, state):
        return self.actor_model.predict([[state]])[0]

    def action_expected_reward(self, states_ph):
        actions = self.actor_model(states_ph)
        q_values = self.critic_model([states_ph, actions])
        return tf.reduce_mean(q_values)

    def next_action(self, state):
        return self.batch_next_action(np.array([state]))[0]

    def batch_next_action(self, states):
        return self.actor_model.predict(states)

    def train(self, state, action, reward, next_state, is_done):
        self.train_on_batch(
            *map(lambda x: np.array([x]), [state, action, reward, next_state, is_done]))

    def train_on_batch(self, states, actions, rewards, next_states, is_dones):
        _, _, loss = sess.run([self.train_critic_step, self.train_actor_step, self.loss], {
            self.state_ph: states, self.actions_ph: actions,
            self.rewards_ph: rewards, self.next_states_ph: next_states, self.is_done_ph: is_dones
        })
        return loss

    def _train_critic(self, state, action, reward, next_state, is_done):
        """For testing purposes"""
        self._train_critic_batch(
            *map(lambda x: np.array([x]), [state, action, reward, next_state, is_done]))

    def _train_critic_batch(self, states, actions, rewards, next_states, is_dones):
        """For testing purposes"""
        sess.run([self.train_critic_step], {
            self.state_ph: states, self.actions_ph: actions,
            self.rewards_ph: rewards, self.next_states_ph: next_states, self.is_done_ph: is_dones
        })

    def _train_actor(self, state, action, reward, next_state, is_done):
        """For testing purposes"""
        self._train_actor_batch(
            *map(lambda x: np.array([x]), [state, action, reward, next_state, is_done]))

    def _train_actor_batch(self, states, actions, rewards, next_states, is_dones):
        """For testing purposes"""
        sess.run([self.train_actor_step], {
            self.state_ph: states, self.actions_ph: actions,
            self.rewards_ph: rewards, self.next_states_ph: next_states, self.is_done_ph: is_dones
        })

    def update_target_network(self):
        assert self.use_target_network
        model_weights = self.critic_model.get_weights()
        target_weights = self.target_network_critic.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * \
                self.tau + target_weights[i]*(1-self.tau)
        self.target_network_critic.set_weights(target_weights)

        model_weights = self.actor_model.get_weights()
        target_weights = self.target_network_actor.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = model_weights[i] * \
                self.tau + target_weights[i]*(1-self.tau)
        self.target_network_actor.set_weights(target_weights)


if __name__ == "__main__":
    def merge_weights(weights):
        return np.hstack([w.flatten() for w in weights])

    obs_size = (2,)
    action_size = 3
    agent = DDPGAgent(obs_size, action_size)

    q_a = agent.get_qa([1, 1], [1, 1, 1])
    print(agent.get_action([1, 1]))
    # Train agent on terminal state
    for i in range(1000):
        agent.train([1, 1], [1, 1, 1], 6, [2, 1], True)
    q_a = agent.get_qa([1, 1], [1, 1, 1])
    assert agent.get_qa([1, 1], [1, 1, 1]) - \
        6 < 0.0001, "Should converge in terminal state"
    assert np.all(agent.get_action([1, 1]) -
                  1 < 0.0001), "Should converge in optimal action"
    print(agent.get_action([1, 1]))

    # Test model training
    original_critic = merge_weights(agent.critic_model.get_weights())
    original_actor = merge_weights(agent.actor_model.get_weights())
    agent._train_actor([1, 1], [1, 1, 1], 6, [2, 1], True)
    updated_critic = merge_weights(agent.critic_model.get_weights())
    updated_actor = merge_weights(agent.actor_model.get_weights())
    assert not np.array_equal(original_actor,
                              updated_actor), "Actor should be updated"

    np.testing.assert_array_equal(original_critic,
                                  updated_critic)
    assert np.array_equal(original_critic,
                          updated_critic), "Critic should not be updated"

    original_critic = merge_weights(agent.critic_model.get_weights())
    original_actor = merge_weights(agent.actor_model.get_weights())
    agent._train_critic([1, 0], [1, 1, 0], 6, [2, 1], True)
    updated_critic = merge_weights(agent.critic_model.get_weights())
    updated_actor = merge_weights(agent.actor_model.get_weights())
    assert not np.array_equal(original_critic,
                              updated_critic), "Critic should be updated"
    assert np.array_equal(original_actor,
                          updated_actor), "Only critic should be updated"
