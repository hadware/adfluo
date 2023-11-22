# TODO
from typing import List, Any

from adfluo import Extractor, Input, F
from adfluo.dataset import ListLoader
from adfluo.processors import DatasetAggregator, DSFeat

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


# TODO:
#  - Test for regular feats after an extractor
#  - Test for Agg() func wrapper
