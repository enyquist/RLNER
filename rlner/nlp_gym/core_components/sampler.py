# standard libaries
from collections import deque
from typing import Any, List

# third party libraries
import numpy as np


class PrioritySampler:
    """Priority Sampler"""

    def __init__(self, max_size: int = None, priority_scale: float = 0.0):
        """
        Creates a priority sampler
        Args:
            max_size (int): maximum size of the queue
            priority_scale (float): 0.0 is a pure uniform sampling, 1.0 is completely priority sampling
        """
        self.max_size = max_size
        self.items = deque(maxlen=self.max_size)
        self.item_priorities = deque(maxlen=self.max_size)
        self.priority_scale = priority_scale

    def add(self, item: Any, priority: float):
        """Add sample"""
        self.items.append(item)
        self.item_priorities.append(priority)

    def sample(self, size: int) -> List[Any]:
        """Get samples"""
        min_sample_size = min(len(self.items), size)
        scaled_item_priorities = np.array(self.item_priorities) ** self.priority_scale
        sample_probs = scaled_item_priorities / np.sum(scaled_item_priorities)
        samples = np.random.choice(a=self.items, p=sample_probs, size=min_sample_size)
        return samples

    def update(self, item: Any, priority: float):
        """Update samples"""
        index = self.items.index(item)
        del self.items[index]
        del self.item_priorities[index]
        self.add(item, priority)

    def get_all_samples(self) -> List[Any]:
        """Get all samples"""
        return self.items


if __name__ == "__main__":
    # standard libaries
    import timeit

    for n_samples in range(1000, 500000, 5000):
        sampler = PrioritySampler(max_size=None, priority_scale=1.0)
        print(f"With {n_samples} samples")
        for i in range(n_samples):
            sampler.add(np.random.rand(), np.random.rand())

        start = timeit.default_timer()
        sampler.sample(1)
        stop = timeit.default_timer()
        print("Execution time " + str(stop - start))
