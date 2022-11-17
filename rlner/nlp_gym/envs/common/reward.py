# standard libaries
from abc import ABC, abstractclassmethod
from typing import List

# rlner libraries
from rlner.nlp_gym.envs.common.observation import BaseObservation


class RewardFunction(ABC):
    """ABC Class for reward functions"""

    @abstractclassmethod
    def __call__(self, observation: BaseObservation, action: str, targets: List[str]) -> float:
        """Call
        Args:
            observation (Observation): current observation at t
            action (str): current action at t
            targets (List[str]): targets of the current sample
        Returns:
            - a scalar reward
        """
        raise NotImplementedError
