from abc import ABC, abstractmethod
from typing import Iterable, List, Dict, Any, Set


class Sample(ABC):
    _features: Dict[str, Any] = {}
    _dropped_features: Set[str] = set()

    def __getitem__(self, item):
        if not isinstance(item, str):
            raise ValueError("Key has to be a string")

        try:
            return self.get_data(item)
        except KeyError:
            pass

        try:
            return self.get_feature(item)
        except KeyError:
            raise KeyError("Can't find data nor feature with such a name")

    @property
    @abstractmethod
    def id(self):
        pass

    def __hash__(self):
        return hash(self.id)

    def get_data(self, data_name: str):
        """This method should be overriden. Depending on the data that is being
        'asked', returns the appropriate `
        SampleData` instance or a raw value"""
        raise NotImplementedError()

    def get_feature(self, feature_name: str):
        return self._features[feature_name]

    # TODO: figure out behavior with feature-storage objects
    def store_feature(self, name, feature, drop_on_save=False):
        """Stores the feature
        In the future, this may cache 'heavy' features in a H5 file to
        prevent overloading the memory"""
        self._features[name] = feature
        if drop_on_save:
            self._dropped_features.add(name)


class DictSample(Sample):

    def __init__(self, sample_dict: Dict[str, Any], sample_id: int):
        self.sample_id = sample_dict.get("id", str(sample_id))
        self.sample_dict = sample_dict

    @property
    def id(self):
        return self.sample_id

    def get_data(self, data_name: str):
        return self.sample_dict[data_name]


class DatasetLoader(ABC):
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


class ListLoader(DatasetLoader):

    def __init__(self, samples: List[Dict, Sample]):
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
