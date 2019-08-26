import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Set


class Sample(ABC):
    def __init__(self):
        self.features: Dict[str, Any] = {}
        self.labels: Dict[str, Any] = {}
        self.dropped_features: Set[str] = set()
        self.dropped_labels: Set[str] = set()

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

    def get_data(self, data_name: str):
        """This method should be overriden. Depending on the data that is being
        'asked', returns the appropriate `
        SampleData` instance or a raw value"""
        raise NotImplementedError()

    def get_feature(self, feature_name: str):
        return self.features[feature_name]

    def store_feature(self, name, feature, drop_on_save=False):
        """Stores the feature
        In the future, this may cache 'heavy' features in a H5 file to
        prevent overloading the memory"""
        self.features[name] = feature
        if drop_on_save:
            self.dropped_features.add(name)

    def store_label(self, name, label, drop_on_save=False):
        self.labels[name] = label
        if drop_on_save:
            self.dropped_labels.add(label)

    def to_h5(self, h5_file):
        """Writes the extracted labels and features to the h5 file"""
        raise NotImplemented()

    def to_pickle(self):
        """Formats the sample's feature to a dictionary that can then
        be pickled"""

        def to_storable(data_dict):
            d = {}
            for data_name, sample_data in data_dict.items():
                if isinstance(sample_data, SampleData):
                    d[data_name] = sample_data.to_storable()
                else:
                    d[data_name] = sample_data
            return d
        saved_features = {feat_name: data
                          for feat_name, data in self.features.items()
                          if feat_name not in self.dropped_features}
        saved_labels = {label_name: data
                        for label_name, data in self.labels.items()
                        if label_name not in self.dropped_labels}
        return {
            "features": to_storable(saved_features),
            "labels": to_storable(saved_labels)
        }


class SampleData:
    def to_storable(self):
        raise NotImplementedError()
