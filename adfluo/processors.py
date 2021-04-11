import sys
from abc import ABC, abstractmethod
from dis import get_instructions
from inspect import signature
from typing import Any, List, Tuple, Callable, TYPE_CHECKING, Hashable, Optional, Union

from sortedcontainers import SortedDict

from .dataset import Sample
from .utils import logger, extraction_policy

if TYPE_CHECKING:
    from .pipeline import PipelineElement


class ProcessorParameter:
    def __init__(self, value: Hashable):
        self.value: Hashable = value


def param(default: Optional[Hashable] = None) -> Any:
    return ProcessorParameter(default)


class BaseProcessor(ABC):
    """Base class for a processor from the feature extraction pipeline"""

    def __init__(self, **kwargs):
        param_names = set(k for k, v in self.__class__.__dict__.items()
                          if isinstance(v, ProcessorParameter))
        for key, val in kwargs.items():
            if key not in param_names:
                raise AttributeError(f"Attribute {key} isn't a processor parameter")
            setattr(self, key, val)

        self.post_init()

    def post_init(self):
        pass

    @property
    def current_sample(self):
        return self._current_sample

    @property
    def nb_args(self):
        return len(signature(self.process).parameters)

    @abstractmethod
    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    @property
    def _params(self) -> SortedDict:
        param_dict = SortedDict()
        for k, v in self.__class__.__dict__.items():
            if isinstance(v, ProcessorParameter):
                param_dict[k] = v
        for k, v in self.__dict__.items():
            if isinstance(v, ProcessorParameter):
                param_dict[k] = v
        return param_dict

    def __setattr__(self, key, value):
        try:
            # TODO: check that processor parameter has a value when instanciated
            attribute = super().__getattribute__(key)
            if isinstance(attribute, ProcessorParameter):
                attribute.value = value
        except AttributeError:
            super().__setattr__(key, value)

    def __getattribute__(self, item):
        attribute = super().__getattribute__(key)
        if isinstance(attribute, ProcessorParameter):
            return attribute.value
        else:
            return attribute

    def __hash__(self):
        return hash((self.__class__, tuple(self._params.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __rshift__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.append(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.concatenate(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline

    def __add__(self, other: 'PipelineElement'):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.merge_proc(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.merge_pipeline(other)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return new_pipeline


class SampleProcessor(BaseProcessor):
    """Processes one sample after the other, independently"""

    def __call__(self, sample: Sample, sample_data: Tuple[Any]) -> Any:

        # trying to process the sample. If an error is raised, two
        # possible outcomes:
        # - if we have to fail on error, than the error is raised
        # (with a cleaner call stack)
        # - if not, we just replace the processed output with None
        # and print an error message
        try:
            self._current_sample = sample
            processed_sample = self.process(*sample_data)
        except Exception as e:
            if not extraction_policy.skip_errors:
                tb = sys.exc_info()[2]
                raise type(e)(("In processor %s, on sample %s : "
                               % (repr(self), sample.id)) +
                              str(e)).with_traceback(tb)
            else:
                logger.warning("Got error in processor %s on sample %s : %s" %
                               (type(self).__name__, sample.id, str(e)))
                return None, None
        else:
            return sample, processed_sample


class FunctionWrapperMixin:

    def __init__(self, fun: Callable):
        if len(signature(fun).parameters) != 1:
            raise ValueError("Function must have one and only one parameter")
        self.fun = fun

    @property
    def nb_args(self):
        return len(signature(self.fun).parameters)

    def __hash__(self):
        """Hashes the disassembled code of the wrapped function."""
        instructions = tuple((instr.opname, instr.arg, instr.argval)
                             for instr in get_instructions(self.fun))
        return hash(instructions)


class FunctionWrapperProcessor(SampleProcessor, FunctionWrapperMixin):
    """Used to wrap simple functions that can be used inline, without
    a processor"""

    def __repr__(self):
        return f"{self.fun.__name__}"

    def process(self, *args):
        return self.fun(*args)


F = FunctionWrapperProcessor


class BatchProcessor(SampleProcessor):
    """Processor class that requires the full list of samples from the
    dataset to be able to process individual samples"""

    @abstractmethod
    def full_dataset_process(self, samples_data: List[Union[Any, Tuple]]):
        """Processes the full dataset of samples. Doesn't return anything,
        store the results as instance attributes"""
        pass


class SampleInputProcessor(SampleProcessor):
    """Processor that pulls data from samples."""
    input: str = param()

    def __init__(self, input: str):
        super().__init__(input=input)

    def process(self, sample: Sample) -> Any:
        return sample[self.input]


Input = SampleInputProcessor


class PrinterProcessor(SampleProcessor):
    name: str = param()

    def __init__(self, name: Optional[str] = None):
        super().__init__(name=name)

    def process(self, *args) -> Tuple:
        print(f"{self.name} received {args} ")
        return args


Printer = PrinterProcessor()


class SampleFeatureProcessor(SampleProcessor):
    """A subtype of `PassThroughProcessor` used to reference a feature"""
    feat_name: str = param()

    def __init__(self, feat_name: str):
        super().__init__(feat_name=feat_name)

    def process(self, *args) -> Tuple:
        return args


Feat = SampleFeatureProcessor


class DatasetAggregator(BaseProcessor):

    @abstractmethod
    def aggregate(self, samples_data: List[Union[Any, Tuple]]) -> Any:
        pass

    def process(self, *args) -> Any:
        pass

    def __call__(self, *args, **kwargs):
        pass  # TODO


class FunctionWrapperAggregator(DatasetAggregator, FunctionWrapperMixin):

    def __repr__(self):
        return f"Aggregator({self.fun.__name__})"

    def aggregate(self, samples_data: List[Union[Any, Tuple]]) -> Any:
        return self.fun(samples_data)


Aggregator = FunctionWrapperAggregator
