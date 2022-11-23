# standard libaries
from pathlib import Path
from typing import List

# third party libraries
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras

# rlner libraries
from rlner.reinforce.model import PolicyNet


class Agent:
    """REINFORCE agent"""

    def __init__(self, action_dim: int = 1):
        """Intialize Agent"""
        self.policy_net = PolicyNet(action_dim=action_dim)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7)
        self.gamma = 0.99

    def dump(self, filepath: Path) -> None:
        """Save an agent's policy

        Args:
            filepath (Path): path to model weights
        """
        self.policy_net.save(filepath)

    def load(self, filepath: Path) -> PolicyNet:
        """Load an agent's policy from file

        Args:
            filepath (Path): path to model weights

        Returns:
            Agent: Poliy Net
        """
        self.policy_net = keras.models.load_model(filepath)

    def policy(self, observation: np.ndarray) -> tf.Tensor:
        """Apply agent policy to an observation

        Args:
            observation (np.ndarray): Vectorized observation

        Returns:
            tf.Tensor: Action
        """
        observation = observation.reshape(1, -1)
        observation = tf.convert_to_tensor(observation, dtype=tf.float32)
        action_logits = self.policy_net(observation)
        action = tf.random.categorical(tf.math.log(action_logits), num_samples=1)
        return action

    def get_action(self, observation: np.ndarray) -> int:
        """Get an action from an agent for interacting with the environment

        Args:
            observation (np.ndarray): Vectorized observation

        Returns:
            int: Action
        """
        action = self.policy(observation).numpy()
        return int(action.squeeze())

    def learn(self, states: List[np.ndarray], rewards: List[float], actions: List[int]) -> None:
        """Update agent policy

        Args:
            states (List[np.ndarray]): States
            rewards (List[float]): Rewards
            actions (List[int]): actions
        """
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

    def loss(self, action_probabilities: tf.Tensor, action: int, reward: float) -> tf.Tensor:
        """Log-loss function

        Args:
            action_probabilities (tf.Tensor): Action Distribution
            action (int): action taken
            reward (float): reward recieved

        Returns:
            tf.Tensor: Log-Loss
        """
        dist = tfp.distributions.Categorical(probs=action_probabilities, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        loss = -log_prob * reward
        return loss
