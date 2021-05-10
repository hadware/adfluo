import pytest

from adfluo import SampleProcessor, param, Input, Feat, F, Sample
from adfluo.exceptions import DuplicateSampleError
from adfluo.extractor import Extractor

dataset = [{
    "data_a": i,
    "data_b": chr(i)
} for i in range(50)]

features = [
    {"times_two": i * 2,
     "times_three": i * 3,
     "times_two_plus_one": i * 2 + 1,
     "combined": i + (i * 2 + 1),
     "combined_plus_one": (i + (i * 2 + 1)) + 1
     } for i in range(50)

]


class TimesX(SampleProcessor):
    factor = param(default=2)

    def process(self, n: int) -> int:
        return self.factor * n


def ordinal(char: str) -> int:
    return ord(char)


def create_dag(extractor: Extractor):
    extractor.add_extraction(Input("data_a") >> TimesX() >> Feat("times_two"))
    extractor.add_extraction(Input("data_a") >> TimesX(factor=3) >> Feat("times_three"))
    extractor.add_extraction(Input("data_a")
                             >> TimesX()
                             >> F(lambda x: x + 1)
                             >> Feat("times_two_plus_one"))
    extractor.add_extraction((Feat("times_two_plus_one")
                              + (Input("data_b") >> F(ordinal)))
                             >> F(lambda x, y: x + y)
                             >> Feat("combined"))
    extractor.add_extraction((Feat("times_two_plus_one")
                              + (Input("data_b") >> F(ordinal)))
                             >> F(lambda x, y: x + y)
                             >> F(lambda x: x + 1) >> Feat("combined_plus_one"))


def test_dag_construction():
    extractor = Extractor()
    create_dag(extractor)
    dag = extractor.extraction_DAG
    assert dag.features == set(features[0].keys())
    assert dag.inputs == {"data_a", "data_b"}


def test_dict_extraction():
    extractor = Extractor(show_progress=False)
    create_dag(extractor)
    d = extractor.extract_to_dict(dataset=dataset, storage_indexing="sample")
    assert d == {str(i): feat_dict for i, feat_dict in enumerate(features)}


def test_dropped_features():
    extractor = Extractor(show_progress=False)
    extractor.add_extraction(Input("data_a") >> TimesX(factor=4) >> Feat("times_four"),
                             drop_on_save=True)
    create_dag(extractor)
    feats_drop = [{k: v for k, v in fdict.items() if k != "times_two"} for fdict in features]
    extractor.dropped_features.add("times_two")
    assert extractor.dropped_features == {"times_two", "times_four"}
    d = extractor.extract_to_dict(dataset)
    assert d == {str(i): feat_dict for i, feat_dict in enumerate(feats_drop)}


def test_extraction_order():
    pass


def test_storage_indexing():
    pass


def test_duplicate_sample():
    class BadDuplicateSample(Sample):

        @property
        def id(self):
            return "4577"

        def __getitem__(self, item):
            if item == "data_a":
                return 1
            elif item =="data_b":
                return "a"

    dataset = [BadDuplicateSample() for i in range(3)]
    extractor = Extractor(show_progress=False)
    create_dag(extractor)
    with pytest.raises(DuplicateSampleError):
        extractor.extract_to_dict(dataset, extraction_order="sample")
