"""

Ring buffers for memory replay.

"""

import numpy as np
import random
from typing import Any, List


class MemoryBuffer:
    " Generalized ring buffer class. "
    def __init__(self, size: int) -> None:
        size += 1
        self.data = [None] * size
        self.first_mem = 0
        self.last_mem = 0
        self.size = size

    def append(self, element: Any) -> None:
        self.data[self.last_mem] = element
        self.last_mem = (self.last_mem + 1) % self.size
        if self.last_mem == self.first_mem:
            self.data[self.first_mem] = None
            self.first_mem = (self.first_mem + 1) % self.size

    def rand_sample(self, sample_size: int = 4) -> List[Any]:
        if self.first_mem < self.last_mem:
            sample_size = min(sample_size, self.last_mem - self.first_mem)
            indexes = np.random.randint(low=self.first_mem, high=self.last_mem, size=sample_size)
        else:
            indexes = random.choices(list(range(0, self.last_mem)) + list(range(self.first_mem + 1, self.size)),
                                     k=sample_size)
        return [self.data[i] for i in indexes]

    def __len__(self) -> int:
        if self.last_mem < self.first_mem:
            return self.last_mem + len(self.data) - self.first_mem
        else:
            return self.last_mem - self.first_mem

    def __iter__(self) -> int:
        for i in range(len(self)):
            yield self[i]


class ShortTermMemoryBuffer(MemoryBuffer):
    " Simple ring buffer to store small data lists, targeted to store four or five board states. "
    def get(self) -> np.array:
        res = []
        s = self.first_mem
        while s != self.last_mem:
            res.append(self.data[s % self.size])
            s = (s + 1) % self.size
        res = np.asarray(res).reshape(-1)
        return res
