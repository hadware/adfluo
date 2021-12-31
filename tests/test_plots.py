import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from adfluo import Input, F, Feat, SampleProcessor
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

