from typing import Tuple

from adfluo import Input, F, Feat, SampleProcessor
from adfluo.extraction_graph import ExtractionDAG
from adfluo.plotting import plot_dag


def test_plot_pipeline():
    class Adder(SampleProcessor):

        def process(self, a: int, b: int):
            return a + b

    def times_two(n: int):
        return n * 2

    def add_one(n: int) -> int:
        return n + 1

    pl = (Input("a") + (Input("b") >> F(add_one))) >> Adder() >> F(times_two) >> Feat("test_feat")
    plot_png = plot_dag(pl, show=True)


def test_plot_graph():
    def a(arg) -> Tuple[str, str]: pass

    def b(arg_a, arg_b) -> float: pass

    def c(arg_a, arg_b) -> ExtractionDAG: pass

    def d(arg) -> "VeryLongClassTypeName": pass

    dag = ExtractionDAG()
    dag.add_pipeline(((Input("input_a") >> F(a)) + Input("input_b")) >> F(b) >> F(d) >> Feat("feat_b"))
    dag.add_pipeline(((Input("input_a") >> F(a)) + Input("input_b")) >> F(c) >> F(d) >> Feat("feat_c"))
    # TODO: investigate why there isn't any F(c) in the nodes
    pl = ((Input("input_a") >> F(a)) + Input("input_b")) >> F(c) >> F(d) >> Feat("feat_c")
    plot_png = plot_dag(dag, show=True)
