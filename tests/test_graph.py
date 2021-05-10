import pytest

from adfluo.dataset import DictSample
from adfluo.extraction_graph import ExtractionDAG, SampleProcessorNode
from adfluo.processors import Input, F, Feat

def test_two_pipeline_parallel():
    def a(arg): pass

    def b(arg): pass

    def c(arg): pass

    def d(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("input_b") >> F(a) >> F(b) >> Feat("feat_b"))
    dag.add_pipeline(Input("input_d") >> F(c) >> F(d) >> Feat("feat_d"))
    assert dag.features == {"feat_b", "feat_d"}
    assert dag.inputs == {"input_d", "input_b"}
    assert dag.feature_nodes["feat_b"].parents[0].processor == F(b)
    assert dag.feature_nodes["feat_d"].parents[0].processor == F(c)


def test_two_pipelines_simple_merge():
    def a(arg): pass

    def b(arg): pass

    def c(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(a) >> F(b) >> Feat("feat_b"))
    dag.add_pipeline(Input("test_input") >> F(a) >> F(c) >> Feat("feat_c"))
    assert dag.features == {"feat_b", "feat_c"}
    assert dag.inputs == {"test_input"}
    assert dag.feature_nodes["feat_b"].parents[0].processor == F(b)
    assert dag.feature_nodes["feat_c"].parents[0].processor == F(c)
    assert (dag.feature_nodes["feat_b"].parents[0].parents[0].processor
            ==
            dag.feature_nodes["feat_c"].parents[0].parents[0].processor
            ==
            F(a))


def test_three_pipelines_simple_merge():
    def a(arg): pass

    def b(arg): pass

    def c(arg): pass

    def d(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(a) >> F(b) >> Feat("feat_b"))
    dag.add_pipeline(Input("test_input") >> F(a) >> F(c) >> Feat("feat_c"))
    dag.add_pipeline(Input("test_input") >> F(a) >> F(d) >> Feat("feat_d"))
    assert dag.features == {"feat_b", "feat_c", "feat_d"}
    assert dag.inputs == {"test_input"}
    assert dag.feature_nodes["feat_b"].parents[0].processor == F(b)
    assert dag.feature_nodes["feat_c"].parents[0].processor == F(c)
    assert dag.feature_nodes["feat_d"].parents[0].processor == F(d)
    assert (dag.feature_nodes["feat_b"].parents[0].parents[0].processor
            ==
            dag.feature_nodes["feat_c"].parents[0].parents[0].processor
            ==
            dag.feature_nodes["feat_d"].parents[0].parents[0].processor
            ==
            F(a))


def test_branches_merge():
    def a(arg): pass

    def b(arg_a, arg_b): pass

    def c(arg_a, arg_b): pass

    def d(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline((Input("input_a") >> F(a) + Input("input_b")) >> F(b) >> F(d) >> Feat("feat_b"))
    dag.add_pipeline((Input("input_a") >> F(a) + Input("input_b")) >> F(c) >> F(d) >> Feat("feat_c"))
    assert dag.features == {"feat_b", "feat_c"}
    assert dag.inputs == {"input_a", "input_b"}
    assert dag.feature_nodes["feat_b"].parents[0].processor == F(d)
    assert dag.feature_nodes["feat_c"].parents[0].processor == F(d)
    assert dag.feature_nodes["feat_b"].parents[0].parents[0].processor == F(b)
    assert dag.feature_nodes["feat_c"].parents[0].parents[0].processor == F(c)
    assert (len(dag.feature_nodes["feat_b"].parents[0].parents[0].parents)
            ==
            len(dag.feature_nodes["feat_c"].parents[0].parents[0].parents)
            == 2)
    assert (dag.feature_nodes["feat_b"].parents[0].parents[0].parents
            ==
            dag.feature_nodes["feat_c"].parents[0].parents[0].parents)


def test_dependency_solving():
    def a(arg): pass

    def b(arg_a, arg_b): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(a) >> Feat("feat_a"))
    dag.add_pipeline((Input("test_input") + Feat("feat_a")) >> F(b) >> Feat("feat_b"))
    dag.solve_dependencies()
    assert dag.features == {"feat_b", "feat_a"}
    assert dag.inputs == {"test_input"}
    assert dag.feature_nodes["feat_a"].children[0].processor == F(b)
    assert dag.feature_nodes["feat_b"].parents[0].processor == F(b)


def test_unsolvable_feature():
    def a(arg): pass

    def b(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(a) >> Feat("feat_a"))
    dag.add_pipeline(Feat("feat_c") >> F(b) >> Feat("feat_b"))
    with pytest.raises(ValueError):
        dag.solve_dependencies()


def test_feature_already_in_graph():
    def a(arg): pass

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(a) >> Feat("feat_a"))
    with pytest.raises(AssertionError):
        dag.add_pipeline(Input("test_other_input") >> F(a) >> Feat("feat_a"))


def test_caching_simple():
    def add_one(n: int):
        return n + 1

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(add_one) >> Feat("feat_a"))
    dag.add_pipeline(Input("test_input") >> F(add_one) >> Feat("feat_b"))
    sample = DictSample({"test_input": 1}, 0)
    assert dag.feature_nodes["feat_a"].parents[0] == dag.feature_nodes["feat_b"].parents[0]
    cached_node: SampleProcessorNode = dag.feature_nodes["feat_a"].parents[0]
    assert len(cached_node.children) == 2
    out = dag.feature_nodes["feat_a"][sample]
    assert out == 2
    assert len(cached_node._samples_cache) == 1
    assert cached_node._samples_cache["0"] == 2
    assert cached_node._samples_cache_hits["0"] == 1

    out = dag.feature_nodes["feat_a"][sample]
    assert out == 2
    assert len(cached_node._samples_cache) == 0


def test_caching_advanced():
    def add_one(n: int):
        return n + 1

    dag = ExtractionDAG()
    dag.add_pipeline(Input("test_input") >> F(add_one) >> Feat("feat_a"))
    dag.add_pipeline(Input("test_input") >> F(add_one) >> Feat("feat_b"))
    dag.add_pipeline(Input("test_input") >> F(add_one) >> F(add_one) >> Feat("feat_c"))
    dag.add_pipeline(Input("test_input") >> F(add_one) >> F(add_one) >> Feat("feat_d"))
    sample_a = DictSample({"test_input": 1}, 0)
    sample_b = DictSample({"test_input": 2}, 1)
    assert dag.feature_nodes["feat_a"].parents[0] == dag.feature_nodes["feat_b"].parents[0]
    assert dag.feature_nodes["feat_c"].parents[0] == dag.feature_nodes["feat_d"].parents[0]
    cached_node: SampleProcessorNode = dag.feature_nodes["feat_a"].parents[0]
    assert len(cached_node.children) == 3
    out = dag.feature_nodes["feat_a"][sample_a]
    assert out == 2
    out = dag.feature_nodes["feat_a"][sample_b]
    assert out == 3
    assert len(cached_node._samples_cache) == 2
    assert cached_node._samples_cache["0"] == 2
    assert cached_node._samples_cache["1"] == 3
    assert cached_node._samples_cache_hits["0"] == 1

    out = dag.feature_nodes["feat_b"][sample_a]
    assert cached_node._samples_cache_hits["0"] == 2

# TODO: make a cache test where the sample processors are checking that they shouldn't be called more
#  than once

def test_error_management():
    # todo : check that skip_error raises a badsample
    # todo : check that skip_error=false raises the right exception
    pass