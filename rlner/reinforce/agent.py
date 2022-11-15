# third party libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# rlner libraries
from rlner.reinforce.model import PolicyNet


class Agent:
    """REINFORCE agent"""

    def __init__(self, action_dim: int = 1):
        """Intialize Agent"""
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.gamma = 0.99

    def policy(self, observation):
        """Apply policy to observation"""
        observation = observation.reshape(1, -1)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_logits = self.policy_net(observation)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        return action

    def get_action(self, observation):
        """Get action from policy"""
        action = self.policy(observation).numpy()
        return action.squeeze()

    def learn(self, states, rewards, actions):
        """Training Loop"""
        discounted_reward = 0
        discounted_rewards = []
        rewards.reverse()
        for r in rewards:
            discounted_reward = r + self.gamma * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards.reverse()

        for state, reward, action in zip(states, discounted_rewards, actions):
            with tf.GradientTape() as tape:
                action_probabilities = self.policy_net(np.array([state]), training=True)
                loss = self.loss(action_probabilities, action, reward)
            grads = tape.gradient(loss, self.policy_net.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))

    def loss(self, action_probabilities, action, reward):
        """Log-Loss function"""
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss
