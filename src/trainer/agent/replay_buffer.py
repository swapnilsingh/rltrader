import random
from collections import deque
from typing import Tuple

class ReplayBuffer:
    def __init__(self, max_size=1000, strategy="static"):
        """
        Args:
            max_size (int): Maximum buffer size.
            strategy (str): One of ['static', 'train_once', 'dynamic'].
        """
        self.strategy = strategy
        self.max_size = max_size
        self.buffer = [] if strategy == "train_once" else deque(maxlen=max_size)

    def add(self, experience: Tuple):
        """
        Adds a new experience to the buffer. Only valid 5-tuples are allowed:
        (state, action, reward, next_state, quantity_class)
        """
        if not isinstance(experience, tuple) or len(experience) != 5:
            print(f"⚠️ ReplayBuffer: Skipping malformed experience: {experience}")
            return
        self.buffer.append(experience)

    def sample(self, batch_size: int):
        """
        Returns a list of valid 5-tuples. Skips malformed entries defensively.
        """
        valid_batch = [e for e in self.buffer if isinstance(e, tuple) and len(e) == 5]
        if not valid_batch:
            print("⚠️ ReplayBuffer: No valid experiences available for sampling.")
            return []

        if self.strategy == "train_once":
            sample_size = min(batch_size, len(valid_batch))
            batch = valid_batch[:sample_size]
            self.buffer = valid_batch[sample_size:]  # drop used
            return batch
        else:
            return random.sample(valid_batch, min(batch_size, len(valid_batch)))

    def resize(self, new_size: int):
        """
        Dynamically resize the buffer (only for 'static' or 'dynamic' strategies).
        """
        if self.strategy == "train_once":
            return
        old = list(self.buffer)
        self.max_size = new_size
        self.buffer = deque(old[-new_size:], maxlen=new_size)

    def clear(self):
        """
        Clears all experiences from the buffer.
        """
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
