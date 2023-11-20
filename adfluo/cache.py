from abc import ABCMeta, abstractmethod
from typing import Dict, Set

from .dataset import Sample
from .exceptions import BadSampleException
from .types import SampleData, SampleID


class BaseCache(metaclass=ABCMeta):

    @abstractmethod
    def __setitem__(self, sample: Sample, value: SampleData):
        pass

    @abstractmethod
    def __getitem__(self, sample: Sample) -> SampleData:
        pass

    def add_failed_sample(self, sample: Sample):
        pass

    @abstractmethod
    def reset(self):
        pass




class SampleCache(BaseCache):

    def __init__(self):
        self._samples_cache: Dict[SampleID, Any] = dict()
        self._samples_cache_hits: Dict[SampleID, int] = dict()
        self._failed_samples: Set[SampleID] = set()

    def __setitem__(self, sample: Sample, data: SampleData):
        self._samples_cache[sample.id] = data
        self._samples_cache_hits[sample.id] = 1

    def __getitem__(self, sample: Sample) -> SampleData:
        if sample.id in self._failed_samples:
            raise BadSampleException(sample)

        if sample.id in self._samples_cache:
            # retrieving the sample and incrementing the cache hits counter
            cached_output = self._samples_cache[sample.id]
            self._samples_cache_hits[sample.id] += 1
            # if the cache hits equals the number of children, the sample's
            # value can be dropped from the cache
            if self._samples_cache_hits[sample.id] >= len(self.children):
                del self._samples_cache[sample.id]
            return cached_output
        else:
            raise KeyError("Sample not in cache")

    def add_failed_sample(self, sample: Sample):
        self._failed_samples.add(sample.id)

    def reset(self):
        self._samples_cache = dict()
        self._samples_cache_hits = dict()
        self._failed_samples: Set[SampleID] = set()


class SingleValueCache(BaseCache):

    def __init__(self):
        self._cached_value = None
        self._has_been_cached = False

    def __setitem__(self, sample: Sample, value: SampleData):
        self._cached_value = value
        self._has_been_cached = True

    def __getitem__(self, sample: Sample) -> SampleData:
        if self._has_been_cached:
            raise KeyError("Sample not in cache")
        else:
            return self._cached_value

    def reset(self):
        self._has_been_cached = False
        self._cached_value = None
