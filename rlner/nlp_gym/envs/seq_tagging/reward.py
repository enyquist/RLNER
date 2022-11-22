# standard libaries
import copy
from pathlib import Path
from typing import List

# third party libraries
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# rlner libraries
from rlner.nlp_gym.envs.common.observation import BaseObservation
from rlner.nlp_gym.envs.common.reward import RewardFunction
from rlner.nlp_gym.metrics.seq_tag import EntityScores

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
EMBEDDING_DIR = DATA_DIR / "embeddings"
PREPARED_DIR = DATA_DIR / "prepared"
NOISE_DIR = PREPARED_DIR / "noise"
LOG_DIR = ROOT_DIR / "models/logs/"


class EntityF1Score(RewardFunction):
    """
    Computes micro f1 score between predicted and target as the final reward
    """

    def __init__(self, dense: bool, average: str):
        """Init"""
        self.dense = dense
        self.average = average
        self.reward_fn = self._dense if self.dense else self._sparse

    def _sparse(self, targets: List[str], prev_action_history: List[str], current_action_history: List[str]) -> float:
        if len(current_action_history) >= len(targets):
            reward = EntityScores(average=self.average)(targets, current_action_history)["f1"]
        else:
            reward = 0.0
        return reward

    def _dense(self, targets: List[str], prev_action_history: List[str], current_action_history: List[str]) -> float:
        # we compute reward as the change in the score
        # step reward as change in the scores
        # as good actions lead to increase in the scores
        previous_score = EntityScores(average=self.average)(targets[: len(prev_action_history)], prev_action_history)[
            "f1"
        ]
        current_score = EntityScores(average=self.average)(
            targets[: len(current_action_history)], current_action_history
        )["f1"]
        reward = current_score - previous_score
        return reward

    def __call__(self, observation: BaseObservation, action: str, targets: List[str], text: List[str] = None) -> float:
        """Call"""
        # get previous and current actions
        prev_action_history = observation.get_current_action_history()
        current_action_history = copy.deepcopy(observation.get_current_action_history())
        current_action_history.append(action)
        reward = self.reward_fn(targets, prev_action_history, current_action_history)
        return reward


class BiLSTMLossScore(RewardFunction):
    """
    Computes the Bi-LSTM-CRF validation loss between predicted and target as the final reward
    """

    def __init__(self, model, val_ds, max_len, words2index, tags2index):
        """Init"""
        self.reward_fn = self._val_loss
        self.model = model
        self.val_ds = val_ds
        self.max_len = max_len
        self.words2index = words2index
        self.tags2index = tags2index

    def _val_loss(self, observation: BaseObservation, action: str, targets: List[str], text: List[str]) -> float:

        tags = copy.deepcopy(targets)
        obs_idx = observation.current_input_index
        tags[obs_idx] = action

        y = [[self.tags2index.get(w, self.tags2index.get("O")) for w in s] for s in [tags]]
        y = pad_sequences(maxlen=self.max_len, sequences=y, padding="post", value=self.tags2index["O"])
        X = [[self.words2index.get(w, self.words2index.get("PADword")) for w in s] for s in [text]]
        X = pad_sequences(maxlen=self.max_len, sequences=X, padding="post", value=self.words2index["PADword"])

        x_tensor = tf.convert_to_tensor(X)
        y_tensor = tf.convert_to_tensor(y)

        train_ds = tf.data.Dataset.from_tensor_slices((x_tensor, y_tensor))

        train_ds.batch(1)

        history = self.model.fit(
            train_ds,
            epochs=1,
            verbose=1,
            batch_size=1,
            validation_data=self.val_ds,
        )

        return history.history["val_loss"][-1]

    def __call__(self, observation: BaseObservation, action: str, targets: List[str], text: List[str] = None) -> float:
        """Call"""
        reward = self.reward_fn(observation, action, targets, text)
        return reward
