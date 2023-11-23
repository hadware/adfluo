# TODO
import re
from typing import List, Any

import pytest

from adfluo import Extractor, Input, F, Agg, Feat
from adfluo.dataset import ListLoader
from adfluo.processors import DatasetAggregator, DSFeat, Pass, DSInput

dataset = [{
    "data_a": i,
    "data_b": chr(i)
} for i in range(10)]


def test_simple_aggregation():
    class SumAggregator(DatasetAggregator):
        def aggregate(self, samples_data: List[int]) -> Any:
            return sum(samples_data)

    dataloader = ListLoader(dataset)
    extractor = Extractor()
    extractor.add_extraction(Input("data_a")
                             >> F(lambda x: x + 1)
                             >> SumAggregator()
                             >> DSFeat("sum"))
    out = extractor.extract_aggregations(dataloader)
    assert out["sum"] == 55


def test_agg_wrapper():
    dataloader = ListLoader(dataset)
    extractor = Extractor()
    extractor.add_extraction(Input("data_a")
                             >> F(lambda x: x + 1)
                             >> Agg(lambda x: sum(x))
                             >> DSFeat("sum"))
    out = extractor.extract_aggregations(dataloader)
    assert out["sum"] == 55


def test_regular_feat_after_agg():
    dataloader = ListLoader(dataset)
    extractor = Extractor()
    extractor.add_extraction(
        Input("data_a")
        >> (Agg(lambda x: max(x)) | Pass)
        >> F(lambda m, x: x / m)
        >> Feat("normed"))
    out = extractor.extract_to_dict(dataloader)
    assert out["9"]["normed"] == 1
    assert out["0"]["normed"] == 0


def test_dataset_input():
    class TestLoader(ListLoader):

        def __getitem__(self, data_name):
            if data_name == "max":
                return 9

    dataloader = TestLoader(dataset)
    extractor = Extractor()
    extractor.add_extraction(
        (DSInput("max") | Input("data_a"))
        >> F(lambda m, x: x / m)
        >> Feat("normed"))
    out = extractor.extract_to_dict(dataloader)
    assert out["9"]["normed"] == 1
    assert out["0"]["normed"] == 0


def test_wrong_feat_after_agg():
    extractor = Extractor()
    with pytest.raises(AssertionError,
                       match=re.escape("Cannot have a sample feature Feat(sum) after an aggregation or dataset input")):
        extractor.add_extraction(
            Input("data_a")
            >> Agg(lambda x: sum(x))
            >> F(lambda x: x + 1)
            >> Feat("sum"))
