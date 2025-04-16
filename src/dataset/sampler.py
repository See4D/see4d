import torch
from torch.utils.data import Sampler
from typing import Iterable, Iterator, Optional, Sized, List
## import concatdataset
from torch.utils.data import ConcatDataset
import ipdb
import bisect
import random

from torch.utils.data.dataset import Dataset

class MultiNomialRandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: List[Sized], replacement: bool = True,
                 num_samples: Optional[int] = None, generator=None, p=None, main_process=False) -> None:
        # ipdb.set_trace()
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self.len_per_dataset = [len(dataset) for dataset in data_source]
        self.total_len = sum(self.len_per_dataset)
        
        if p is not None and p[0] is not None:
            sump = sum(p)
            # for i in range(len(p)):
            #     print(f"Dataset {i}: len: {self.len_per_dataset[i]} Estimated samples of prob: {p[i]/sump * self.len_per_dataset[i]:.1f}")
            assert len(p) == len(data_source)
            probs = []
            for _p, dataset_samples in zip(p, self.len_per_dataset):
                probs.extend([_p]*dataset_samples)
            
            average_len = [_p/sump * dataset_samples for _p, dataset_samples in zip(p, self.len_per_dataset)]
            if main_process:
                for i in range(len(p)):
                    print(f"Dataset {i}: chunks/videos: {self.len_per_dataset[i]} Estimated samples of chunks: {average_len[i]:.2f}")
            
        else:
            probs = [1] * self.total_len
        ## probs not need to normalize, it will normalize in torch.multinomial
        self.probs = torch.tensor(probs, dtype=torch.float32)
                
        
        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")
        
        if not self.replacement:
            raise "In this case, datasets are independent, so replacement should be True."
        if num_samples is not None:
            raise NotImplementedError("num_samples is not supported yet.")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return self.total_len
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = self.total_len
        
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
        random_sample_number = 128 # increase this number from 32 to 1024, try to avoid reading same data together
        if self.replacement:
            for _ in range(self.num_samples // random_sample_number):
                yield from torch.multinomial(self.probs, num_samples=random_sample_number, replacement=False, generator=generator).tolist()
                # yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            # yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.multinomial(self.probs, num_samples=self.num_samples % random_sample_number, replacement=False, generator=generator).tolist()
            
        else:
            for _ in range(self.num_samples // n):
                yield from torch.multinomial(self.probs, n, replacement=False, generator=generator).tolist()
                # torch.randperm(n, generator=generator).tolist()

    def __len__(self) -> int:
        return self.num_samples
    

class ConcatDatasetWithIndex(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset]) -> None:
        super().__init__(datasets)
        # import ipdb; ipdb.set_trace()
    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        
        # data_sample = self.datasets[dataset_idx][sample_idx]
        data_sample = self.datasets[dataset_idx][sample_idx]
        if data_sample is None:
            data_sample = self.datasets[dataset_idx][random.randint(0, len(self.datasets[dataset_idx])-1)]
        data_sample['dataset_idx'] = dataset_idx
        return data_sample