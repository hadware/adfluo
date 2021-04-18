from typing import Any

import pytest

from adfluo import ExtractionPipeline
from adfluo.processors import SampleProcessor, F, Input, Feat


def test_processor_chain():
    class A(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class B(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class C(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    a, b, c = A(), B(), C()
    pipeline = a >> b >> c
    assert isinstance(pipeline, ExtractionPipeline)
    assert pipeline.inputs[0].processor is a
    assert pipeline.outputs[0].processor is c
    assert pipeline.inputs[0].children[0].processor is b
    assert pipeline.outputs[0].parents[0].processor is b


def test_proc_merge():
    class A(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class B(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    a, b = A(), B()
    a_b = a + b
    assert isinstance(a_b, ExtractionPipeline)
    assert len(a_b.outputs) == len(a_b.inputs) == 2
    assert a_b.outputs[0].processor is a_b.inputs[0].processor is a
    assert a_b.outputs[1].processor is a_b.inputs[1].processor is b


def test_proc_merge_and_concatenate():
    class A(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class B(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class C(SampleProcessor):
        def process(self, arg_a, arg_b) -> Any:
            pass

    a, b, c = A(), B(), C()

    pipeline = (a + b) >> c
    assert isinstance(pipeline, ExtractionPipeline)
    assert len(pipeline.inputs) == 2
    assert len(pipeline.outputs) == 1
    assert pipeline.inputs[0].processor is a
    assert pipeline.inputs[1].processor is b
    assert pipeline.outputs[0].processor is c
    assert len(pipeline.outputs[0].parents) == 2


def test_proc_concatenate_merge_and_concatenate():
    class A(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class B(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class C(SampleProcessor):
        def process(self, arg_a, arg_b) -> Any:
            pass

    class D(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    a, b, c, d = A(), B(), C(), D()

    pipeline = (d >> (a + b) >> c)
    assert isinstance(pipeline, ExtractionPipeline)
    assert len(pipeline.inputs) == 1
    assert len(pipeline.outputs) == 1
    assert pipeline.inputs[0].processor is d
    assert pipeline.outputs[0].processor is c
    assert len(pipeline.inputs[0].children) == 2
    assert pipeline.inputs[0].children[0].processor is a
    assert pipeline.inputs[0].children[1].processor is b


def test_fun_merge_and_concatenate():
    def a(arg):
        pass

    def b(arg):
        pass

    c = lambda x, y: "4577"

    pipeline = (F(a) + F(b)) >> F(c)
    assert isinstance(pipeline, ExtractionPipeline)
    assert len(pipeline.inputs) == 2
    assert len(pipeline.outputs) == 1
    assert pipeline.inputs[0].processor == F(a)
    assert pipeline.inputs[1].processor == F(b)
    assert pipeline.outputs[0].processor == F(c)
    assert len(pipeline.outputs[0].parents) == 2


def test_pipeline_io_checking():
    class A(SampleProcessor):
        def process(self, *args) -> Any:
            pass

    class B(SampleProcessor):
        def process(self, arg_a, arg_b) -> Any:
            pass

    # checking good pipelines
    pl_good = Input("test_input") >> A() >> A() >> Feat("test_feat")
    pl_good.check()
    pl_good_two_inputs = ((Input("test_input") >> A()) + Input("other")) >> B() >> Feat("test_feat")
    pl_good_two_inputs.check()
    pl_good_two_feats = Input("test_input") >> A() >> A() >> (Feat("test_feat") + (A() >> Feat("other")))
    pl_good_two_feats.check()

    # checking good pipelines
    pl_bad_input = (Input("test_input") + A()) >> B() >> Feat("test_feat")
    with pytest.raises(AssertionError):
        pl_bad_input.check()
    pl_bad_input = A() >> A() >> Feat("test_feat")
    with pytest.raises(AssertionError):
        pl_bad_input.check()
    pl_bad_middle = Input("test_input") >> A() >> Input("wrong") >> A() >> Feat("test_feat")
    with pytest.raises(AssertionError):
        pl_bad_middle.check()
    pl_bad_feat = Input("test_input") >> A() >> A() >> A()
    with pytest.raises(AssertionError):
        pl_bad_feat.check()
    pl_bad_feat = Input("test_input") >> A() >> A() >> (A() + Feat("test_feat"))
    with pytest.raises(AssertionError):
        pl_bad_feat.check()
# TODO : check (a >> (b + c) + d ) >> e
