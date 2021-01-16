import sys
from abc import ABC, abstractmethod
from dis import get_instructions
from inspect import signature
from typing import Any, List, Tuple, Callable, TYPE_CHECKING

from .samples import Sample
from .utils import logger

if TYPE_CHECKING:
    from .pipeline import PipelineElement


class BaseProcessor(ABC):
    """Baseclass for a processor from the feature extraction pipeline"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # ordering the kwargs by name to ensure a consistent hash value
        self.ordered_kwargs = tuple(sorted(list(kwargs.items()),
                                           key=lambda x: x[0]))
        self._current_sample: Sample = None

    @property
    def current_sample(self):
        return self._current_sample

    @current_sample.setter
    def current_sample(self, sample: Sample):
        self._current_sample = sample

    @property
    def nb_args(self):
        return len(signature(self.process).parameters)

    def process(self, *args) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    def __hash__(self):
        return hash((self.__class__, self.ordered_kwargs))

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __repr__(self):
        if self.kwargs:
            return (self.__class__.__name__ +
                    "(%s)" % ",".join("%s=%s" % (key, val)
                                      for key, val in self.ordered_kwargs))
        else:
            return self.__class__.__name__

    def __rshift__(self, other: PipelineElement):
        from .pipeline import (T, Feat, ExtractionPipeline, PIPELINE_TYPE_ERROR,
                               PipelineBuildError)
        if isinstance(other, (BaseProcessor, T, Feat)):
            return ExtractionPipeline([self, other])
        elif callable(other):
            return ExtractionPipeline([self, FunctionWrapperProcessor(other)])
        elif isinstance(other, ExtractionPipeline):
            other.elements.insert(0, self)
        else:
            raise PipelineBuildError(PIPELINE_TYPE_ERROR.format(obj_type=type(other)))

    @abstractmethod
    def __call__(self, sample: Sample, sample_data: Tuple[Any], fail_on_error: bool) -> Any:
        pass


class SampleProcessor(BaseProcessor):
    """Processes one sample after the other, independently"""

    def __call__(self, sample: Sample, sample_data: Tuple[Any], fail_on_error: bool) -> Any:

        # TODO: if there is an error and fail_on_error is false,
        #  maybe set the sample to None instead of the value

        # if the current sample being processed is None, the processor
        # acts as a passthrough
        if sample_data is None:
            return None

        # trying to process the sample. If an error is raised, two
        # possible outcomes:
        # - if we have to fail on error, than the error is raised
        # (with a cleaner call stack)
        # - if not, we just replace the processed output with None
        # and print an error message
        try:
            self.current_sample = sample
            processed_sample = self.process(*sample_data)
        except Exception as e:
            if fail_on_error:
                tb = sys.exc_info()[2]
                raise type(e)(("In processor %s, on sample %s : "
                               % (repr(self), sample.id)) +
                              str(e)).with_traceback(tb)
            else:
                logger.warning("Got error in processor %s on sample %s : %s" %
                               (type(self).__name__, sample.id, str(e)))
                return None
        else:
            return processed_sample


class FunctionWrapperProcessor(SampleProcessor):
    """Used to wrap simple functions that can be used inline, without
    a processor"""

    def __init__(self, fun: Callable):
        super().__init__()
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

    def __repr__(self):
        return f"{self.__class__.__name__}({self.fun.__name__})"

    def process(self, *args):
        return self.fun(*args)


class BatchProcessor(SampleProcessor):
    """Processor class that requires the full list of samples from the
    dataset to be able to process individual samples"""

    @abstractmethod
    def full_dataset_process(self, samples: List[Sample],
                             samples_data: List[Any, Tuple]):
        """Processes the full dataset of samples. Doesn't return anything,
        store the results as instance attributes"""
        pass

class SampleInputProcessor(SampleProcessor):
    """Processor that pulls data from samples."""

    def __init__(self, input: str):
        super().__init__(input=input)
        self.input = input

    def __hash__(self):
        return hash(self.input)

    def process(self, sample: Sample) -> Any:
        try:
            return sample.get_feature(feature_name=self.input)
        except KeyError:
            return sample.get_data(data_name=self.input)


Input = SampleInputProcessor


class PassThroughProcessor(SampleProcessor):
    """Processor that acts as a passthrough, doesn't do anything"""

    def __hash__(self):
        return hash(self.__class__)

    def process(self, *args) -> Tuple:
        return args


Pass = PassThroughProcessor()