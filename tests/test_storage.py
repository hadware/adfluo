import json
import pickle
from io import StringIO, BytesIO
from itertools import chain
from pathlib import Path
from typing import Any

from adfluo import Extractor, Input, Feat, F
from adfluo.dataset import ListLoader
from adfluo.storage import BaseStorage, CSVStorage, JSONStorage, PickleStorage, SplitPickleStorage, StorageProtocol
from adfluo.types import SampleID, FeatureName

DATA_FOLDER = Path(__file__).parent / Path("data/")

samples = [{"feat_a": i, "feat_b": i ** 2} for i in range(3)]
dataset = ListLoader(samples)
feat_c = {sample.id: i * 3 for i, sample in enumerate(dataset)}
feat_d = {sample.id: i - 1 for i, sample in enumerate(dataset)}

storage_dict_sample_idx = {'0': {'feat_a': 0, 'feat_b': 0, 'feat_c': 0},
                           '1': {'feat_a': 1, 'feat_b': 1, 'feat_c': 3},
                           '2': {'feat_a': 2, 'feat_b': 4, 'feat_c': 6}}

storage_dict_feat_idx = {'feat_a': {'0': 0, '1': 1, '2': 2},
                         'feat_b': {'0': 0, '1': 1, '2': 4},
                         'feat_c': {'0': 0, '1': 3, '2': 6}}


def fill_storage(storage: BaseStorage):
    for sample, sample_data in zip(dataset, samples):
        storage.store_sample(sample, sample_data)
    storage.store_feat("feat_c", feat_c)


def test_base_storage():
    storage = BaseStorage(indexing="sample")
    fill_storage(storage)
    assert storage.get_data() == storage_dict_sample_idx

    storage = BaseStorage(indexing="feature")
    fill_storage(storage)
    assert storage.get_data() == storage_dict_feat_idx


def test_csv_storage():
    str_io = StringIO()
    storage = CSVStorage(indexing="sample", output_file=str_io)
    fill_storage(storage)
    storage.write()
    str_io.seek(0)
    with open(DATA_FOLDER / Path("sample_idx.csv")) as csv_file:
        assert str_io.read().split() == csv_file.read().split()

    str_io = StringIO()
    storage = CSVStorage(indexing="feature", output_file=str_io)
    fill_storage(storage)
    storage.write()
    str_io.seek(0)
    with open(DATA_FOLDER / Path("feat_idx.csv")) as csv_file:
        assert str_io.read().split() == csv_file.read().split()


def test_pickle_storage():
    bytes_io = BytesIO()
    storage = PickleStorage(indexing="sample", output_file=bytes_io)
    fill_storage(storage)
    storage.write()
    bytes_io.seek(0)
    assert pickle.load(bytes_io) == storage_dict_sample_idx

    bytes_io = BytesIO()
    storage = PickleStorage(indexing="feature", output_file=bytes_io)
    fill_storage(storage)
    storage.write()
    bytes_io.seek(0)
    assert pickle.load(bytes_io) == storage_dict_feat_idx


def test_json_storage():
    str_io = StringIO()
    storage = JSONStorage(indexing="sample", output_file=str_io)
    fill_storage(storage)
    storage.write()
    str_io.seek(0)
    with open(DATA_FOLDER / Path("sample_idx.json")) as json_file:
        assert json.load(str_io) == json.load(json_file)

    str_io = StringIO()
    storage = JSONStorage(indexing="feature", output_file=str_io)
    fill_storage(storage)
    storage.write()
    str_io.seek(0)
    with open(DATA_FOLDER / Path("feat_idx.json")) as json_file:
        assert json.load(str_io) == json.load(json_file)


# TODO : add test for pickle-per-file storage

def test_pickle_per_file_sample(tmpdir):
    tmpdir = Path(tmpdir.strpath)
    storage = SplitPickleStorage(indexing="sample", output_folder=tmpdir, streaming=False)
    fill_storage(storage)
    storage.write()
    for f in tmpdir.iterdir():
        f.match(r".*\.pckl")
    assert {f.stem for f in tmpdir.iterdir()} == {sample.id for sample in dataset}


def test_pickle_per_file_feature(tmpdir):
    tmpdir = Path(tmpdir.strpath)
    storage = SplitPickleStorage(indexing="feature", output_folder=tmpdir, streaming=False)
    fill_storage(storage)
    storage.write()
    for f in tmpdir.iterdir():
        f.match(r".*\.pckl")
    assert {f.stem for f in tmpdir.iterdir()} == set(storage_dict_feat_idx.keys())


def test_pickle_per_file_streaming_sample(tmpdir):
    tmpdir = Path(tmpdir.strpath)
    storage = SplitPickleStorage(indexing="sample", output_folder=tmpdir, streaming=True)
    samples_ids = set()
    for sample, sample_data in zip(dataset, samples):
        samples_ids.add(sample.id)
        storage.store_sample(sample, sample_data)
        for f in tmpdir.iterdir():
            f.match(r".*\.pckl")
        print({f.stem for f in tmpdir.iterdir()})
        assert {f.stem for f in tmpdir.iterdir()} == samples_ids
        assert storage._data == dict()
    assert {f.stem for f in tmpdir.iterdir()} == {sample.id for sample in dataset}


def test_custom_storage_feature():
    class TestStorage(StorageProtocol):

        def __init__(self):
            self.values = dict()

        def store(self, sample_id: SampleID, feat: FeatureName, value: Any):
            self.values[(sample_id, feat)] = value

    storage = TestStorage()
    samples = [{"input_a": i, "input_b": i ** 2} for i in range(3)]
    dataset = ListLoader(samples)
    extractor = Extractor()
    extractor.add_extraction(Input("input_a") >> Feat("feat_a"))
    extractor.add_extraction(Input("input_b") >> Feat("feat_b", storage=storage))
    output = extractor.extract_to_dict(dataset)
    # testing that custom storage feat isn't in the output anymore
    assert set(chain.from_iterable(feats.keys() for feats in output.values())) == {"feat_a"}
    # testing that the storage worked properly
    assert set(storage.values.keys()) == {(str(i), "feat_b") for i in range(3)}


def test_storage_flatten_features():
    samples = [{"input_a": i} for i in range(3)]
    dataset = ListLoader(samples)
    extractor = Extractor()
    extractor.add_extraction(Input("input_a")
                             >> F(lambda i: {"squared": i ** 2, "mult": {"double": 2 * i, "triple": 3 * i}})
                             >> Feat("feat_a"))

    output = extractor.extract_to_dict(dataset, storage_indexing="feature", extraction_order="sample",
                                       flatten_features=True)
    assert set(output.keys()) == {"feat_a_squared", "feat_a_mult_double", "feat_a_mult_triple"}
    assert output["feat_a_mult_triple"]["1"] == 3

    output = extractor.extract_to_dict(dataset, storage_indexing="feature", extraction_order="feature",
                                       flatten_features=True)
    assert set(output.keys()) == {"feat_a_squared", "feat_a_mult_double", "feat_a_mult_triple"}
    assert output["feat_a_mult_triple"]["1"] == 3
