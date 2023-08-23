from typing import Any

import pytest
from sortedcontainers import SortedDict

from adfluo.dataset import DictSample
from adfluo.processors import param, SampleProcessor, F, Input, Feat, ListWrapperProcessor, hparam


def test_proc_params():
    class TestProc(SampleProcessor):
        a = param()
        b = param()

        def process(self, *args) -> Any:
            pass

    assert TestProc(a=1, b="b")._sorted_params == SortedDict({'a': 1, 'b': 'b'})
    assert TestProc(a=1, b=2) == TestProc(a=1, b=2)
    assert TestProc(a=1, b="a") != TestProc(a=1, b="b")
    assert repr(TestProc(a=1, b=2)) == "<TestProc(a=1,b=2)>"

    with pytest.raises(AttributeError, match="Attribute c isn't a processor parameter"):
        TestProc(c=2)


def test_processor_default_params():
    class TestProc(SampleProcessor):
        a = param(1)
        b = param("c")

        def process(self, *args) -> Any:
            pass

    assert TestProc()._sorted_params == SortedDict({'a': 1, 'b': 'c'})
    assert repr(TestProc()) == "<TestProc(a=1,b='c')>"
    assert TestProc(a=2)._sorted_params == SortedDict({'a': 2, 'b': 'c'})


def test_fun_hash():
    def a(param):
        return param * 2

    def c(param):
        return param * 2

    def b(param):
        return param + 2

    assert F(a) == F(a)
    assert F(a) != F(c)
    assert F(a) != F(b)


def test_lambda():
    assert F(lambda x, y: list([x, y])) == F(lambda x, y: list([x, y]))
    assert F(lambda x, y: list([x, y])) != F(lambda x, y: tuple([x, y]))


def test_nb_args():
    def f(a, b):
        return a * b

    def g(a):
        return a ** 2

    def h():
        return "la menuiserie mec"

    assert F(f).nb_args == 2
    assert F(g).nb_args == 1

    with pytest.raises(ValueError, match="Function must have at least one parameter"):
        F(h)


def test_input_proc():
    assert Input(data_name="test") == Input(data_name="test")
    input_proc = Input(data_name="test")
    sample = DictSample({"test": "a", "la_menuiserie": 4577}, sample_id=1)
    assert input_proc(sample, (sample,)) == "a"
    menuiserie_proc = Input(data_name="la_menuiserie")
    assert menuiserie_proc(sample, (sample,)) == 4577


def test_feat_proc():
    feat_proc = Feat(feat_name="test_feat")
    assert feat_proc(None, ("test",)) == "test"


def test_proc_args():
    class PassProc(SampleProcessor):
        def process(self, *args) -> Any:
            return args

    assert PassProc()(None, (1, "a")) == (1, "a")
    assert PassProc()(None, (1, "a", 2)) == (1, "a", 2)

    class SumProc(SampleProcessor):
        def process(self, a, b) -> Any:
            return a + b

    assert SumProc()(None, (1, 2)) == 3


def test_list_processors():
    def f(a: int) -> int:
        return a ** 2

    class SquareProc(SampleProcessor):

        def process(self, a: int) -> int:
            return a ** 2

    assert ListWrapperProcessor(F(f))(None, ([1, 2, 3],)) == [1, 4, 9]
    assert ListWrapperProcessor(SquareProc())(None, ([1, 2, 3],)) == [1, 4, 9]
    assert ListWrapperProcessor(F(f)) == ListWrapperProcessor(F(f))


def test_processor_hparam_hash():
    class ProcA(SampleProcessor):
        param_a: str = param()
        param_b: int = param()

        def process(self, *args) -> Any:
            return 1

    proc_a = ProcA(param_a="test", param_b=hparam("hparam_a"))
    proc_b = ProcA(param_a="test", param_b=hparam("hparam_a"))
    proc_c = ProcA(param_a="test", param_b=hparam("hparam_b"))
    proc_d = ProcA(param_a=hparam("hparam_a"), param_b=hparam("hparam_b"))
    assert proc_a == proc_b
    assert proc_a != proc_c
    assert proc_a != proc_d


def test_processor_hparam_list():
    class ProcA(SampleProcessor):
        param_a: str = param()
        param_b: int = param()
        param_c: int = param()

        def process(self, *args) -> Any:
            return 1

    proc_a = ProcA(param_a=hparam("hparam_a"), param_b=2, param_c=hparam("hparam_b"))
    assert proc_a.hparams == {"hparam_a", "hparam_b"}
    proc_a = ProcA(param_a=hparam("hparam_a"), param_b=2, param_c=2)
    assert proc_a.hparams == {"hparam_a"}


def test_processor_hparam_set():
    class ProcA(SampleProcessor):
        param_a: str = param()
        param_b: int = param()

        def process(self, *args) -> Any:
            return 1

    proc_a = ProcA(param_a="test", param_b=hparam("hparam_b"))
    proc_a.set_hparams(hparam_a=10, hparam_b=34)
    assert proc_a.param_a == "test"
    assert proc_a.param_b == 34

# TODO : unittest BatchProcessors
