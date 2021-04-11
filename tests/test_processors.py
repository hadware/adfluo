from typing import Any

from adfluo.processors import param, SampleProcessor, FunctionWrapperProcessor
from dataclasses import dataclass


def test_proc_params():
    class TestProc(SampleProcessor):
        def __init__(self, a, b):
            self.a = param(a)
            self.b = param(b)

        def process(self, *args) -> Any:
            pass

    assert TestProc(1, 2) == TestProc(1, 2)
    assert TestProc(1, "a") != TestProc(1, "b")
    assert repr(TestProc(1, 2)) == "TestProc(a=1,b=2)"


def test_dataclass_params():
    @dataclass
    class TestProc(SampleProcessor):
        a: int
        b: str

        def process(self, *args) -> Any:
            pass

    assert TestProc(1, "a") == TestProc(1, "a")
    assert TestProc(a=1, b="a") != TestProc(1, "b")
    print(TestProc(1,"a"))


def test_fun_hash():
    def a(param):
        return param * 2

    def c(foo):
        return foo * 2

    def b(param):
        return param + 2

    assert FunctionWrapperProcessor(a) == FunctionWrapperProcessor(a)
    assert FunctionWrapperProcessor(a) == FunctionWrapperProcessor(c)
    assert FunctionWrapperProcessor(a) != FunctionWrapperProcessor(b)


def test_nb_args():
    def f(a, b):
        return a * b
