# third party libraries
from tensorflow import keras
from tensorflow.keras import layers


class PolicyNet(keras.Model):
    """Neural Net for REINFORCE Agent"""

    def __init__(self, action_dim: int = 1):
        """Initialize Network"""
        super().__init__()
        self.fc1 = layers.Dense(24, activation="relu")
        self.fc2 = layers.Dense(36, activation="relu")
        self.fc3 = layers.Dense(action_dim, activation="softmax")

    def call(self, x):
        """Forward Pass"""
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def process(self, observations):
        """Process Batch"""
        action_probabilities = self.predict_on_batch(observations)
        return action_probabilities
