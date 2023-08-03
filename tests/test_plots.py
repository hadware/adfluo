from itertools import product
from numbers import Number
from typing import Tuple

import pytest

from adfluo import Input, F, Feat, SampleProcessor, param
from adfluo.extraction_graph import ExtractionDAG
from adfluo.plotting import plot_dag


@pytest.mark.skip(reason="Plotting is broken")
def test_plot_pipeline():
    class Adder(SampleProcessor):

        def process(self, a: int, b: int):
            return a + b

    def times_two(n: int):
        return n * 2

    def add_one(n: int) -> int:
        return n + 1

    pl = (Input("a") | (Input("b") >> F(add_one))) >> Adder() >> F(times_two) >> Feat("test_feat")
    plot_png = plot_dag(pl, show=True)


@pytest.mark.skip(reason="Plotting is broken")
def test_plot_graph():
    def a(arg) -> Tuple[str, str]: pass

    def b(arg_a, arg_b) -> float: pass

    def c(arg_a, arg_b) -> ExtractionDAG: pass

    def d(arg) -> "VeryLongClassTypeName": pass

    dag = ExtractionDAG()
    dag.add_pipeline(((Input("input_a") >> F(a)) | Input("input_b")) >> F(b) >> F(d) >> Feat("feat_b"))
    dag.add_pipeline(((Input("input_a") >> F(a)) | Input("input_b")) >> F(c) >> F(d) >> Feat("feat_c"))

    plot_png = plot_dag(dag, show=True)


@pytest.mark.skip(reason="Plotting is broken")
def test_plot_wide_dag():
    def a(arg) -> Tuple[str, str]: pass

    def b(arg_a, arg_b) -> float: pass

    class Adder(SampleProcessor):
        val: int = param()

        def process(self, *args) -> Number:
            pass

    class Multiplier(SampleProcessor):
        val: int = param()

        def process(self, *args) -> Number:
            pass

    dag = ExtractionDAG()
    for feat_id, (input_id, mult_val, add_val) in enumerate(product(range(2), range(3), range(10))):
        dag.add_pipeline(Input(F"input_{input_id}")
                         >> F(a)
                         >> F(b)
                         >> Adder(val=add_val)
                         >> Multiplier(val=mult_val)
                         >> Feat(f"feat_{feat_id}"))

    plot_png = plot_dag(dag, show=True)
