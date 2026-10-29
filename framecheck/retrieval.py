"""Step 1: size-matched retrieval of 10-K sentences from the reference firms.

The pool holds sentences that report one change with a stated percentage, no dollar amount, at most one other
number, and no cause, so that the model sees how a change of this size is worded without anything to copy.
For a change, we take the K sentences in the same direction whose size is closest on a log scale.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RESOURCES = Path(__file__).resolve().parent / "resources"
K = 6


class SizeMatchedRetriever:
    def __init__(self, pool=RESOURCES / "retrieval_pool.jsonl.gz", k=K):
        self.pool = pd.read_json(pool, lines=True)
        self.log_size = np.log(self.pool.value.abs().values)
        self.direction = self.pool.direction.values
        self.k = k

    def retrieve(self, value, k=None):
        """The k pool sentences closest in |log size| to a change of `value` % in the same direction."""
        k = k or self.k
        d = np.abs(self.log_size - np.log(abs(value)))
        same = self.direction == ("up" if value > 0 else "down")
        idx = np.argsort(np.where(same, d, 1e9), kind="stable")[:k]
        return self.pool.sentence.values[idx].tolist()
