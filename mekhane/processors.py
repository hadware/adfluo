import inspect
from abc import ABC, abstractmethod
import sys
from typing import Iterable, Generator, Any, List, Tuple

from .samples import Sample


class BaseProcessor(ABC):
    """Baseclass for a processor from the feature extraction pipeline"""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # ordering the kwargs by name to ensure a consistent hash value
        self.ordered_kwargs = tuple(sorted(list(kwargs.items()),
                                           key=lambda x: x[0]))
        self._current_sample: Sample = None

    def accepts(self, input_type):
        try:
            own_input_type = self.process.__annotations__["sample_data"]
            if isinstance(own_input_type, list):
                return input_type in own_input_type
            elif own_input_type == Any:
                return True
            else:
                return own_input_type == input_type
        except KeyError:
            return True

    def outputs(self):
        try:
            return self.process.__annotations__["return"]
        except ValueError:
            return Any

    @property
    def current_sample(self):
        return self._current_sample

    @current_sample.setter
    def current_sample(self, sample: Sample):
        self.current_sample = sample

    def process(self, sample_data: Any) -> Any:
        """Processes just one sample"""
        raise NotImplemented()

    def __hash__(self):
        return hash((self.__class__, self.ordered_kwargs))

    def __eq__(self, other):
        return (self.__class__, self.kwargs) == (other.__class__, other.kwargs)

    def __repr__(self):
        if self.kwargs:
            return (self.__class__.__name__ +
                    "(%s)" % ",".join("%s=%s" % (key, val)
                                      for key, val in self.ordered_kwargs))
        else:
            return self.__class__.__name__

    @abstractmethod
    def __call__(self, samples_gen: Iterable, fail_on_error: bool) \
            -> Generator[Any, None, None]:
        pass


class SampleProcessor(BaseProcessor):
    """Processes one sample after the other, independently"""

    def __call__(self, samples_it: Iterable, fail_on_error: bool):
        for sample, sample_data in samples_it:
            # if the current sample being processed is None, the processor
            # acts as a passthrough
            if sample_data is None:
                yield sample, None
                continue

            # trying to process the sample. If an error is raised, two
            # possible outcomes:
            # - if we have to fail on error, than the error is raised
            # (with a cleaner call stack)
            # - if not, we just replace the processed output with None
            # and print an error message
            try:
                self.current_sample = sample
                processed_sample = self.process(sample_data)
            except Exception as e:
                if fail_on_error:
                    tb = sys.exc_info()[2]
                    raise type(e)(("In processor %s, on sample %s : "
                                   % (repr(self), sample.id)) +
                                  str(e)).with_traceback(tb)
                else:
                    print("Got error in processor %s on sample %s : %s" %
                          (type(self).__name__, sample.id, str(e)))
                    yield sample, None
            else:
                yield sample, processed_sample


class BatchProcessor(BaseProcessor):
    """Processor class that requires the full list of samples from the
    dataset to be able to process individual samples"""

    @abstractmethod
    def full_dataset_process(self, samples: List[Sample], samples_data: List):
        """Processes the full dataset of samples. Doesn't return anything,
        store the results as instance attributes"""
        pass

    @abstractmethod
    def __call__(self, samples_gen: Iterable, fail_on_error: bool) \
            -> Generator[Any, None, None]:
        samples, samples_data = zip(*samples_gen)
        self.full_dataset_process(samples, samples_data)
        for sample, sample_data in zip(samples, samples_data):
            self.current_sample = sample
            yield sample, self.process(sample_data)


ELEMENTS_ALIASES = {
    "denoiser": None,
}


def processor_factory(element_class, params):
    try:
        element_type = ELEMENTS_ALIASES[element_class]
    except KeyError:
        raise ValueError("Element class %s is not recognized. "
                         "Available element classes are %s" %
                         (element_class, ", ".join(ELEMENTS_ALIASES.keys())))

    class_specs = inspect.getfullargspec(element_type.__init__)
    for param, value in params.items():
        if param not in class_specs.args:
            raise ValueError("%s parameter not recognized for %s element" %
                             (param, element_type.__class__.__name__))

        try:
            if not isinstance(value, class_specs.annotations[param]):
                raise ValueError(
                    "%s parameter is of type %s and "
                    "should be of type %s" % (param, str(
                        type(value)), str(class_specs.annotations[param])))
        except KeyError:
            pass

    return element_type(**params)
