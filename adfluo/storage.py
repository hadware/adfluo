import csv
import pickle
from collections import defaultdict
from csv import Dialect
from typing import Optional, Union, TextIO, Dict, Any, BinaryIO, Literal, Set

from .dataset import Sample

StorageIndexing = Literal["feature", "sample"]
Feature = str
SampleID = Union[str, int]


class BaseStorage:

    def __init__(self, indexing: StorageIndexing):
        self.indexing = indexing
        self._data: Dict[SampleID, Dict[Feature, Any]] = defaultdict(dict)
        self._features: Set[Feature] = set()

    def store_feat(self, feature: str, data: Dict[Sample, Any]):
        self._features.add(feature)
        for sample, value in data.items():
            self._data[sample.id][feature] = value

    def store_sample(self, sample: Sample, data: Dict[Feature, Any]):
        self._features.update(set(data.keys()))
        self._data[sample.id] = data

    def get_value(self):
        if self.indexing == "feature":
            out_data = defaultdict(dict)
            for sample_id, feat_dict in self._data.items():
                for feat, value in feat_dict:
                    out_data[feat][sample_id] = value
        else:
            out_data = self._data
        return dict(out_data)


class CSVStorage(BaseStorage):

    def __init__(self,
                 indexing: StorageIndexing,
                 output_file: TextIO,
                 dialect: Optional[Dialect]):
        super().__init__(indexing)
        self.file = output_file
        if dialect is None:
            self.dialect = Dialect()
        else:
            self.dialect = dialect

    def write(self):
        data = self.get_value()
        if self.indexing == "sample":
            index_column = "sample_id"
            fields = [index_column] + list(self._features)
        else:
            index_column = "feature"
            fields = [index_column] + list(self._data.keys())
        writer = csv.DictWriter(self.file, fieldnames=fields, dialect=self.dialect)
        writer.writeheader()
        for key, data in data.items():
            row_dict = {index_column: key}
            row_dict.update(**data)
            writer.writerow(row_dict)


class PickleStorage(BaseStorage):

    def __init__(self,
                 indexing: StorageIndexing,
                 output_file: BinaryIO):
        super().__init__(indexing)
        self.file = output_file

    def write(self):
        pickle.dump(self.get_value(), self.file)


class DataFrameStorage(BaseStorage):

    def get_value(self):
        data = super().get_value()
        import pandas as pd
        return pd.DataFrame.from_dict(data)


class HDF5Storage(BaseStorage):
    pass
