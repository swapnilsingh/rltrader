import random
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=1000, strategy="static"):
        """
        Args:
            max_size (int): Max buffer size (ignored in train_once mode).
            strategy (str): One of ['static', 'train_once', 'dynamic']
        """
        self.strategy = strategy
        self.max_size = max_size

        if strategy == "train_once":
            self.buffer = []  # Simple list, flushed after use
        else:
            self.buffer = deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        if self.strategy == "train_once":
            sample_size = min(batch_size, len(self.buffer))
            batch = self.buffer[:sample_size]
            self.buffer = self.buffer[sample_size:]  # Drop used samples
            return batch
        else:
            return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def resize(self, new_size):
        if self.strategy == "train_once":
            return  # resizing makes no sense here
        old = list(self.buffer)
        self.max_size = new_size
        self.buffer = deque(old[-new_size:], maxlen=new_size)

    def clear(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)
