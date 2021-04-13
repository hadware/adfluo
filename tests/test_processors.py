from typing import Any

import pytest
from sortedcontainers import SortedDict

from adfluo.processors import param, SampleProcessor, F, F
from dataclasses import dataclass

# TODO : check error raising for non-param arguments in kwargs
def test_proc_params():
    class TestProc(SampleProcessor):
        a = param()
        b = param()

        def process(self, *args) -> Any:
            pass

    assert TestProc(a=1, b="b")._params == SortedDict({'a': 1, 'b': 'b'})
    assert TestProc(a=1, b=2) == TestProc(a=1, b=2)
    assert TestProc(a=1, b="a") != TestProc(a=1, b="b")
    assert repr(TestProc(a=1, b=2)) == "<TestProc(a=1,b=2)>"


def test_processor_default_params():
    class TestProc(SampleProcessor):
        a = param(1)
        b = param("c")

        def process(self, *args) -> Any:
            pass

    assert TestProc()._params == SortedDict({'a': 1, 'b': 'c'})
    assert repr(TestProc()) == "<TestProc(a=1,b='c')>"


def test_fun_hash():
    def a(param):
        return param * 2

    def c(param):
        return param * 2

    def b(param):
        return param + 2

    assert F(a) == F(a)
    assert F(a) == F(c)
    assert F(a) != F(b)


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