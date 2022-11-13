from torch.utils.data import Sampler

from TTS.tts.datasets.dataset import TTSDataset

import random
import numpy as np


class LengthSortSampler(Sampler):

    def __init__(self, data_source: TTSDataset, batch_size):
        super(LengthSortSampler, self).__init__(data_source)

        self._data_source = data_source
        self._lengths = data_source.lengths
        self.batch_size = batch_size

    def __iter__(self):
        # TODO: This module also implements dynamic batch size

        sorted_lengths = np.argsort(self._lengths)
        for i in range(len(sorted_lengths) // self.batch_size):
            start_index = random.randint(0, len(sorted_lengths) - self.batch_size)

            if i == 0:
                start_index = len(sorted_lengths)

            batch_size = self.batch_size
            if start_index > len(sorted_lengths) // 2:
                batch_size = batch_size // 2

            if i == 0:
                start_index = len(sorted_lengths) - batch_size

            yield sorted_lengths[start_index: start_index + batch_size]

    def __len__(self):
        return len(self._data_source) // self.batch_size
