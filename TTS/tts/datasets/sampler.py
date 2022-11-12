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
        sorted_lengths = np.argsort(self._lengths)
        for _ in range(len(self._data_source) // self.batch_size):
            start_index = random.randint(0, len(self._data_source) - self.batch_size)
            yield sorted_lengths[start_index: start_index + self.batch_size]

    def __len__(self):
        return len(self._data_source) // self.batch_size
