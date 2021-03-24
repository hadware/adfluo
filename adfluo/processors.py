import sys
from abc import ABC, abstractmethod
from dataclasses import is_dataclass, fields
from dis import get_instructions
from inspect import signature
from typing import Any, List, Tuple, Callable, TYPE_CHECKING, Hashable, Optional

from sortedcontainers import SortedDict

from .dataset import Sample
from .utils import logger, extraction_policy

if TYPE_CHECKING:
    from .pipeline import PipelineElement


class ProcessorParameter:
    def __init__(self, value: Hashable):
        self.value = value


def param(value: Hashable) -> Any:
    return ProcessorParameter(value)


class BaseProcessor(ABC):
    """Baseclass for a processor from the feature extraction pipeline"""
    _params_dict: SortedDict = SortedDict()
    _current_sample: Sample = None

    @property
    def current_sample(self):
        return self._current_sample

    @property
    def nb_args(self):
        return len(signature(self.process).parameters)

    @property
    def _params(self):
        params = self._params_dict.copy()
        if is_dataclass(self):
            for field in fields(self):
                params[field.name] = self.__getattribute__(field.name)
        return params

    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    def __setattr__(self, key, value):
        if isinstance(value, ProcessorParameter):
            self._params_dict[key] = value.value
        else:
            super().__setattr__(key, value)

    def __getattribute__(self, item):
        return self._params_dict.get(item,
                                     default=super().__getattribute__(item))

    def __hash__(self):
        return hash((self.__class__, tuple(self._params.items())))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        if self._params:
            params = ",".join("%s=%s" % (key, val)
                              for key, val in self._params.items())
            return f"{self.__class__.__name__}({params})"
        else:
            return self.__class__.__name__

    def __rshift__(self, other: PipelineElement):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.append(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.concatenate(other)
        elif callable(other):
            new_pipeline.append(FunctionWrapperProcessor(other))
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self

    def __add__(self, other: PipelineElement):
        from .pipeline import (ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        new_pipeline = ExtractionPipeline()
        new_pipeline.append(self)
        if isinstance(other, BaseProcessor):
            new_pipeline.merge_proc(other)
        elif isinstance(other, ExtractionPipeline):
            new_pipeline.merge_pipeline(other)
        elif callable(other):
            new_pipeline.merge_proc(FunctionWrapperProcessor(other))
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))
        return self


class SampleProcessor(BaseProcessor):
    """Processes one sample after the other, independently"""

    def __call__(self, sample: Sample, sample_data: Tuple[Any]) -> Any:

        # TODO: if there is an error and fail_on_error is false,
        #  maybe set the sample to None instead of the value

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


class BatchProcessor(SampleProcessor):
    """Processor class that requires the full list of samples from the
    dataset to be able to process individual samples"""

    @abstractmethod
    def full_dataset_process(self, samples_data: List[Any, Tuple]):
        """Processes the full dataset of samples. Doesn't return anything,
        store the results as instance attributes"""
        pass


class SampleInputProcessor(SampleProcessor):
    """Processor that pulls data from samples."""

    def __init__(self, input: str, is_feat: bool = False):
        # is_feat is not part of the hash because input data names and features names
        # are part of the same set and are unique in that set
        super().__init__(input=input)
        self.input = input
        self.is_feat = is_feat

    def __hash__(self):
        return hash(self.input)

    def process(self, sample: Sample) -> Any:
        if self.is_feat:
            return sample.get_feature(feature_name=self.input)
        else:
            return sample.get_data(data_name=self.input)


Input = SampleInputProcessor


class PrinterProcessor(SampleProcessor):

    def __init__(self, name: Optional[str] = None):
        self.name = param(name)

    def process(self, *args) -> Tuple:
        print(f"{self.name} received {args} ")
        return args


Printer = PrinterProcessor()


class SampleFeatureProcessor(SampleProcessor):
    """A subtype of `PassThroughProcessor` used to reference a feature"""

    def __init__(self, feat_name: str):
        self.feature = param(feat_name)

    def process(self, *args) -> Tuple:
        return args


Feat = SampleFeatureProcessor


class DatasetAggregator(BaseProcessor):

    @abstractmethod
    def aggregate(self, samples_data: List[Any, Tuple]) -> Any:
        pass

    def process(self, *args) -> Any:
        pass

    def __call__(self, *args, **kwargs):
        pass  # TODO


class FunctionWrapperAggregator(DatasetAggregator, FunctionWrapperMixin):

    def __repr__(self):
        return f"Aggregator({self.fun.__name__})"

    def aggregate(self, samples_data: List[Any, Tuple]) -> Any:
        return self.fun(samples_data)


Aggregator = FunctionWrapperAggregator
