from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Any, Union, Sized

from adfluo.types import SampleID


class Sample(ABC):

    def __new__(cls, *args, **kwargs):
        if not isinstance(cls.id, property):
            raise TypeError("id method has to be a property")
        return super().__new__(cls)

    @property
    @abstractmethod
    def id(self) -> Union[str, int]:
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.id)

    @abstractmethod
    def __getitem__(self, data_name: str) -> Any:
        """This method should be overridden. Depending on the data that is being
        queried, returns the right value"""
        raise NotImplementedError()


class DictSample(Sample):

    def __init__(self, sample_dict: Dict[str, Any], sample_id: int):
        self.sample_id = sample_dict.get("id", str(sample_id))
        self.sample_dict = sample_dict

    @property
    def id(self):
        return self.sample_id

    def __getitem__(self, data_name):
        return self.sample_dict[data_name]


class DatasetLoader(ABC, Iterable, Sized):
    """
    Child classes of this class should take care of loading a dataset
    and formatting it to samples, then storing it into the sample
    attribute.
    """

    @abstractmethod
    def __len__(self):
        raise NotImplementedError()

    @abstractmethod
    def __iter__(self) -> Iterable[Sample]:
        raise NotImplementedError()

    def __getitem__(self, data_name: str) -> Any:
        # TODO: doc
        pass


class ListLoader(DatasetLoader):

    def __init__(self, samples: List[Union[Dict, Sample]]):
        self._samples: List[Sample] = []
        # Wrapping dictionnaries in a sample dict
        for i, sample in enumerate(samples):
            if isinstance(sample, dict):
                sample = DictSample(sample, i)
            self._samples.append(sample)

    def __len__(self):
        return len(self._samples)

    def __iter__(self):
        return iter(self._samples)


class SubsetLoader(DatasetLoader):
    """Dataset wrapper that keeps only a definite subset of the original dataset's
    samples"""

    def __init__(self, dataset: DatasetLoader, kept_samples: List[SampleID]):
        self.dataset = dataset
        self.kept_samples = set(kept_samples)

    def __len__(self):
        return len(self.kept_samples)

    def __iter__(self):
        for sample in self.dataset:
            if sample.id in self.kept_samples:
                yield sample
