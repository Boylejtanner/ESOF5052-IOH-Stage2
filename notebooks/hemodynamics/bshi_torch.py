"""torch side of the BSHI loader (imports torch at module level on purpose: the class must be importable by Windows spawn DataLoader workers)."""
from __future__ import annotations

import torch

from . import bshi as S


class BshiBatches(torch.utils.data.IterableDataset):
    """Yields ready-made batches (dict of tensors). Each DataLoader worker takes every ``num_workers``-th case, so workers never load the same case."""

    def __init__(self, root, index, variant, batch_size, shuffle, seed, cases_per_chunk, include_dataset_flag):
        self.root, self.index, self.variant, self.batch_size = root, index, variant, batch_size
        self.shuffle, self.seed, self.cases_per_chunk, self.include_dataset_flag = shuffle, seed, cases_per_chunk, include_dataset_flag
        self.epoch = 0
        self._calls = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        subset = (info.id, info.num_workers) if info else None
        epoch = self.epoch + self._calls                      # persistent workers keep their own copy: every new iteration is a new epoch with a new shuffle
        self._calls += 1
        for batch in S.window_batch_iter(self.root, self.index, self.variant, self.batch_size, self.shuffle, self.seed + epoch, self.cases_per_chunk,
                                         include_dataset_flag=self.include_dataset_flag, case_subset=subset):
            yield {k: torch.from_numpy(v) for k, v in batch.items()}
