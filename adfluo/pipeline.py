import warnings
from typing import List, Union, Dict, Any

from .cache import SingleValueCache
from .dataset import Sample, DictSample
from .exceptions import PipelineBuildError, PIPELINE_TYPE_ERROR
from .extraction_graph import SampleProcessorNode, FeatureNode, InputNode, FeatureName, AggregatorNode, BaseInputNode, \
    BaseFeatureNode, DatasetInputNode, DatasetFeatureNode
from .plotting import plot_dag
from .processors import ProcessorBase, SampleProcessor, \
    Input, Feat, DatasetAggregator, DSInput, DSFeat

PipelineElement = Union['ExtractionPipeline', ProcessorBase]
ProcessorNode = Union[SampleProcessorNode, AggregatorNode]


def wrap_processor(proc: ProcessorBase) -> ProcessorNode:
    if isinstance(proc, Feat):
        return FeatureNode(proc)
    if isinstance(proc, DSFeat):
        return DatasetFeatureNode(proc)
    elif isinstance(proc, Input):
        return InputNode(proc)
    elif isinstance(proc, DSInput):
        return DatasetInputNode(proc)
    elif isinstance(proc, DatasetAggregator):
        return AggregatorNode(proc)
    elif isinstance(proc, SampleProcessor):
        return SampleProcessorNode(proc)
    else:
        raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(proc)))


# TODO : add "connect" function and streamline some of this code
class ExtractionPipeline:

    def __init__(self):
        self.inputs: List[ProcessorNode] = []
        self.outputs: List[ProcessorNode] = []
        self.all_nodes: List[ProcessorNode] = []

    @property
    def nb_inputs(self):
        return len(self.inputs)

    @property
    def nb_outputs(self):
        return len(self.outputs)

    def check_aggregations(self):
        stack: List[ProcessorNode] = [node for node in self.all_nodes
                                      if isinstance(node, (DatasetInputNode, AggregatorNode))]
        node_added_to_stack = True
        single_value_nodes = []
        while node_added_to_stack:
            node_added_to_stack = False
            node = stack.pop(0)

            assert not isinstance(node, FeatureNode), \
                f"Cannot have a sample feature {str(node)} after an aggregation or dataset input"

            # skipping input nodes/agg nodes
            if isinstance(node, (DatasetInputNode, AggregatorNode)):
                single_value_nodes.append(node)
                stack += node.children
                node_added_to_stack = True
                continue

            # end of branch
            if isinstance(node, DatasetFeatureNode):
                continue

            # merger nodes situation
            if len(node.parents) > 1:
                parents_in_candidates = ((parent_node in single_value_nodes)
                                         for parent_node in node.parents)
                if not all(parents_in_candidates):
                    stack.append(node)
                    continue

            # remaining situation is : we have a node that's a 'single-value' candidate
            # and its cache has to be changed.
            # children nodes can also be added to the stack
            single_value_nodes.append(node)
            # setting cache to single value
            node.cache = SingleValueCache(node)
            stack += node.children
            node_added_to_stack = True

    def check(self):
        """
        Checks that :
         - input nodes are all `InputNode` or `FeatureNode`
         - output nodes are all `FeatureNode`
         - "inner" nodes can't be input/feature nodes
         """

        # TODO : better error
        for node in self.inputs:
            assert isinstance(node, (BaseInputNode, BaseFeatureNode))
        for node in self.outputs:
            assert isinstance(node, BaseFeatureNode)
        ends = set(self.inputs + self.outputs)
        for node in self.all_nodes:
            if node not in ends:
                assert not isinstance(node, (BaseFeatureNode, BaseInputNode))

        # then, checking aggregations if needed
        for node in self.all_nodes:
            if isinstance(node, (DatasetInputNode, AggregatorNode)):
                self.check_aggregations()
                break

    def append(self, proc: ProcessorBase):
        new_node = wrap_processor(proc)
        # extraction DAG has no node: new dag!
        if self.nb_inputs == 0:
            self.inputs = [new_node]

        # "regular" case: adding a node after the last one
        elif self.nb_outputs == 1:
            old_tail = self.outputs[0]
            old_tail.children = [new_node]
            new_node.parents = [old_tail]

        # adding a node as a merger of several branches
        else:
            assert proc.parameters.accept(self.nb_outputs)
            new_node.parents = self.outputs
            for o in self.outputs:
                o.children = [new_node]
        self.outputs = [new_node]

        self.all_nodes.append(new_node)

    def concatenate(self, pipeline: 'ExtractionPipeline'):
        """Appends (in place) another pipeline to the current pipeline instance"""
        if self.nb_outputs == pipeline.nb_inputs:
            for right_out, left_in in zip(self.outputs, pipeline.inputs):
                right_out.children = [left_in]
                left_in.parents = [right_out]

        elif self.nb_outputs == 1 and pipeline.nb_inputs > 1:
            self.outputs[0].children = pipeline.inputs
            for i in pipeline.inputs:
                i.parents = [self.outputs[0]]

        elif pipeline.nb_inputs == 1 and self.nb_outputs > 1:
            # TODO: better error
            assert pipeline.inputs[0].processor.parameters.accept(self.nb_outputs)
            pipeline.inputs[0].parents = self.outputs
            for o in self.outputs:
                o.children = [pipeline.inputs[0]]

        self.outputs = pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def add_parallel_proc(self, proc: ProcessorBase):
        """Adds a new processor that processes in parallel to the current pipeline instance"""
        new_node = wrap_processor(proc)
        self.inputs.append(new_node)
        self.outputs.append(new_node)
        self.all_nodes.append(new_node)

    def add_parallel_pipeline(self, pipeline: 'ExtractionPipeline'):
        """Adds a new pipeline that processes in parallel to the current pipeline instance"""
        self.inputs += pipeline.inputs
        self.outputs += pipeline.outputs
        self.all_nodes += pipeline.all_nodes

    def __rshift__(self, other: PipelineElement):
        if isinstance(other, ProcessorBase):
            self.append(other)
        elif isinstance(other, ExtractionPipeline):
            self.concatenate(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __or__(self, other: PipelineElement):
        if isinstance(other, ProcessorBase):
            self.add_parallel_proc(other)
        elif isinstance(other, ExtractionPipeline):
            self.add_parallel_pipeline(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __call__(self, sample: Union[Sample, Dict[str, Any]]) -> Dict[FeatureName, Any]:
        # TODO: add support for datasets:
        #  - actually any input is a dataset (wrap sample with fake dataset)
        #  - add a rootnode and then remove it (needed for aggs)
        self.check()

        if isinstance(sample, dict):
            sample = DictSample(sample, 0)
        output_dict: Dict[FeatureName, Any] = {}
        # TODO: catch some errors (due to unsolved features/batch proc with no dataset)
        #  and "contextualize" them
        for output_node in self.outputs:
            # skipping nodes that aren't feature nodes
            if not isinstance(output_node, FeatureNode):
                continue
            output_node: FeatureNode
            output_dict[output_node.processor.feat_name] = output_node[sample]
        return output_dict

    def _repr_png_(self):
        """Ipython notebook visualization"""
        try:
            plot_dag(self, show=True)
        except ImportError as err:
            warnings.warn(str(err))
