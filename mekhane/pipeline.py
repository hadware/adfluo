from .helpers import consecutive_couples
from .samples import Sample
from .processors import BaseProcessor, processor_factory
from typing import List, Generator, Any, Iterable, Tuple


class ExtractionPipeline:
    def __init__(self,
                 processors: List[BaseProcessor],
                 fail_on_error: bool = True):
        self.processors = processors
        self.fail_on_error = fail_on_error

    @classmethod
    def from_conf(cls, yaml_elements_list):
        # todo: refactor this to match new
        procs = [
            processor_factory(element_name, params)
            for element_name, params in yaml_elements_list
        ]
        return cls(procs)

    def check_typing(self):
        """Checks if the input and outputs of processors have matching types"""
        for proc_out, proc_in in consecutive_couples(self.processors):
            if not proc_in.accepts(proc_out.outputs()):
                raise TypeError(
                    "Processor %s does not support output type "
                    "from processor %s" % (type(proc_in).__name__,
                                           type(proc_out).__name__))

    def error_catcher(self, proc, chain_call):
        while True:
            try:
                sample, sample_data = next(chain_call)
            except StopIteration:
                break  # Iterator exhausted: stop the loop
            else:
                if isinstance(sample_data, Exception):
                    if not self.fail_on_error:
                        print(
                            "Got error in processor %s on sample %s : %s" %
                            (type(proc).__name__, sample.id, str(sample_data)))
                        yield sample, None
                    else:
                        raise sample_data
                else:
                    yield sample, sample_data

    def _chain_processors(self, samples: Iterable):
        chain_call = samples
        for proc in self.processors:
            chain_call = proc(chain_call, self.fail_on_error)
        return chain_call

    def __call__(self, samples_couples: Iterable[Tuple[Sample, Any]]) \
            -> Generator:
        return self._chain_processors(samples_couples)
