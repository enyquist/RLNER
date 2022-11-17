# standard libaries
from typing import List

# third party libraries
from gym.spaces.discrete import Discrete


class ActionSpace(Discrete):
    """Action Space"""

    def __init__(self, actions: List[str]):
        """Init"""
        self.actions = actions
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}
        super().__init__(len(self.actions))

    def __post_init__(self):
        self._ix_to_action = {ix: action for ix, action in enumerate(self.actions)}
        self._action_to_ix = {action: ix for ix, action in enumerate(self.actions)}

    def action_to_ix(self, action: str) -> int:
        """Map action string to index"""
        return self._action_to_ix[action]

    def ix_to_action(self, ix: int) -> str:
        """Map index to action string"""
        return self._ix_to_action[ix]

    def size(self) -> int:
        """Size of action space"""
        return self.n

    def __repr__(self):
        return f"Discrete Action Space with {self.size()} actions: {self.actions}"
