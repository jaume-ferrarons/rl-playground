from tensorflow.compat.v1.keras import models, layers, backend, optimizers
import tensorflow.compat.v1 as tf
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


print("Tensorflow version:", tf.__version__)

# tf.compat.v1.disable_eager_execution()
tf.reset_default_graph()
sess = tf.InteractiveSession()


class QLearningAgent():
    def __init__(self, obs_size, n_actions,
                 model=None,
                 use_target_network=False,
                 learning_rate=1e-3, reward_discount=0.99):
        self.n_actions = n_actions
        self.lr = learning_rate
        self.rd = reward_discount

        # Init model
        if model is None:
            model = models.Sequential()
            model.add(layers.Dense(20, input_shape=obs_size, activation='elu'))
            model.add(layers.Dense(10, activation='elu'))
            model.add(layers.Dense(n_actions))
            model.summary()
        self.model = model
        self.use_target_network = use_target_network

        self.state_ph = tf.placeholder(tf.float32, shape=(None, *obs_size))
        self.actions_ph = tf.placeholder(tf.int32, shape=(None))
        self.rewards_ph = tf.placeholder(tf.float32, shape=(None))
        self.next_states_ph = tf.placeholder(
            tf.float32, shape=(None, *obs_size))
        self.is_done_ph = tf.placeholder(tf.float32, shape=(None))

        self.trainable_variables = self.model.trainable_variables
        if use_target_network:
            self.target_network = tf.keras.models.clone_model(self.model)

        self.loss = self.lossf(self.state_ph, self.actions_ph,
                               self.rewards_ph, self.next_states_ph, self.is_done_ph)
        self.set_learning_rate(self.lr)
        self.update_target_network()

    def set_learning_rate(self, learning_rate):
        self.train_step = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss,
            var_list=self.trainable_variables
        )
        sess.run(tf.global_variables_initializer())

    def lossf(self, state_ph, actions_ph, rewards_ph, next_states_ph, is_done_ph):
        q_values = self.model(state_ph)
        action_q_value = tf.reduce_sum(
            tf.one_hot(actions_ph, self.n_actions) * q_values, axis=1)
        if self.use_target_network:
            next_q_values = self.target_network(next_states_ph)
        else:
            next_q_values = self.model(next_states_ph)
        next_state_values = tf.reduce_max(next_q_values, axis=1)
        is_not_done = 1 - is_done_ph
        target_qvalues_for_actions = rewards_ph + self.rd*next_state_values*is_not_done
        loss = (action_q_value - tf.stop_gradient(target_qvalues_for_actions))**2
        return tf.reduce_mean(loss)

    def _get_state_qs(self, state):
        return self.model.predict(np.array([state]))[0]

    def get_qa(self, state, action):
        return self._get_state_qs(state)[action]

    def get_max_q(self, state):
        return np.max(self._get_state_qs(state))

    def next_action(self, state, exploration_rate=0.0):
        return self.batch_next_action(np.array([state]), exploration_rate)[0]

    def batch_next_action(self, states, exploration_rate=0.0):
        predictions = self.model.predict(states)
        actions = np.argmax(predictions, axis=1)
        return [np.random.choice(self.n_actions) if np.random.random() < exploration_rate else a for a in actions]

    def train(self, state, action, reward, next_state, is_done):
        self.train_on_batch(
            *map(lambda x: np.array([x]), [state, action, reward, next_state, is_done]))

    def train_on_batch(self, states, actions, rewards, next_states, is_dones):
        _, loss = sess.run([self.train_step, self.loss], {
            self.state_ph: states, self.actions_ph: actions,
            self.rewards_ph: rewards, self.next_states_ph: next_states, self.is_done_ph: is_dones
        })
        return loss

    def update_target_network(self):
        assert self.use_target_network
        self.target_network.set_weights(self.model.get_weights())


if __name__ == "__main__":
    agent = QLearningAgent(obs_size=(1,), n_actions=2, use_target_network=True)

    # Check initial weights of target network are equal
    w1 = agent.model.get_weights()
    w2 = agent.target_network.get_weights()
    assert np.all([np.all(a == b) for (a, b) in zip(w1, w2)]
                  ), "All weights expected to be equal"

    q_a = agent.get_qa([1], 1)
    print(q_a)
    # Train agent on terminal state
    for i in range(1000):
        agent.train([1], 1, 6, [2], True)
    q_a = agent.get_qa([1], 1)
    assert agent.get_qa([1], 1) - \
        6 < 0.0001, "Should converge in terminal state"

    # Train agent on non-terminal state
    for i in range(1000):
        agent.train([4], 0, 2, [0], False)
    q_a = agent.get_qa([4], 0)
    assert agent.get_qa([4], 0) - (agent.get_max_q([0]) *
                                   0.99 + 2) < 0.0001, "Should converge in non-terminal state"

    # Test agent action sampling
    actions = [agent.next_action([1], 0.0) for _ in range(20)]
    assert len(np.unique(actions)) == 1, "Only one action expected when greedy"
    actions = [agent.next_action([1], 1.0) for _ in range(20)]
    assert len(np.unique(actions)
               ) > 1, "More than one action expected when not greedy"

    w1 = agent.model.get_weights()
    w2 = agent.target_network.get_weights()
    assert not np.all([np.all(a == b) for (a, b) in zip(w1, w2)]
                      ), "Weights expected to be different"
    agent.update_target_network()
    w1 = agent.model.get_weights()
    w2 = agent.target_network.get_weights()
    old_target = w2
    assert np.all([np.all(a == b) for (a, b) in zip(w1, w2)]
                  ), "Weights expected to be equal"

    agent.train([1], 1, 6, [2], True)
    w1 = agent.model.get_weights()
    w2 = agent.target_network.get_weights()
    assert not np.all([np.all(a == b) for (a, b) in zip(w1, w2)]
                      ), "Weights expected to be different"
    assert np.all([np.all(a == b) for (a, b) in zip(old_target, w2)]
                  ), "Target should not be updated"
