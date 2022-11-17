# standard libaries
from abc import ABC, abstractmethod
from dataclasses import dataclass

# third party libraries
import torch


@dataclass
class BaseObservation:
    """
    Placeholder for observation data class
    """

    pass


class BaseObservationFeaturizer(ABC):
    """ABC Class for Observation Featurizer"""

    @abstractmethod
    def featurize(self, observation: BaseObservation) -> torch.Tensor:
        """Featurize observations"""
        raise NotImplementedError

    def get_observation_dim(self) -> int:
        """
        Returns the observation dim
        """
        return self.get_input_dim() + self.get_context_dim()
