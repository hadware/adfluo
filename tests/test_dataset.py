from typing import List

import pytest

from adfluo.dataset import Sample, DictSample, DatasetLoader, ListLoader


def test_sample_base():
    class TestSample(Sample):

        def __init__(self, i: int):
            self.i = i

        @property
        def id(self):
            return str(self.i)

        def __getitem__(self, item):
            if item == "idx":
                return self.i
            elif item == "idx_sqr":
                return self.i ** 2

    samples = [TestSample(i) for i in range(10)]
    for i, sample in enumerate(samples):
        assert sample.id == str(i)
        assert sample["idx"] == i
        assert sample["idx_sqr"] == i ** 2

def test_sample_property():
    class MySample(Sample):

        def id(self):
            return "4577"

    with pytest.raises(TypeError, match="id method has to be a property")   :
        sample = MySample()


def test_dict_samples():
    samples = [DictSample({"idx": i, "idx_sqr": i ** 2}, sample_id=i)
               for i in range(10)]
    for i, sample in enumerate(samples):
        assert sample.id == str(i)
        assert sample["idx"] == i
        assert sample["idx_sqr"] == i ** 2


def test_dataset_loader():
    class TestSample(Sample):

        def __init__(self, i: int):
            self.i = i

        @property
        def id(self):
            return str(self.i)

        def __getitem__(self, item):
            if item == "idx":
                return self.i
            elif item == "idx_sqr":
                return self.i ** 2

    class TestDataset(DatasetLoader):
        def __init__(self, samples_list):
            self.sample_list = samples_list

        def __iter__(self):
            return iter(self.sample_list)

        def __len__(self):
            return len(self.sample_list)

    dataset = TestDataset([TestSample(i) for i in range(10)])
    assert len(dataset) == 10
    for i, sample in enumerate(dataset):
        assert sample.id == str(i)
        assert sample["idx"] == i
        assert sample["idx_sqr"] == i ** 2


def test_list_dataset():
    samples = ListLoader([{"idx": i, "idx_sqr": i ** 2}
                          for i in range(10)])
    for i, sample in enumerate(samples):
        assert sample.id == str(i)
        assert sample["idx"] == i
        assert sample["idx_sqr"] == i ** 2
