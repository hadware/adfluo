from enum import Enum
import sys
from collections import OrderedDict
from enum import Enum
from typing import List, Union, Dict

from tqdm import tqdm

from .extraction_tree import ProcessorsTree
from .loader import DatasetLoader
from .pipeline import ExtractionPipeline
from .processors import BaseProcessor



class ExtractionType(Enum):
    FEATURE = 1
    LABEL = 2

    def __str__(self):
        return "feature" if self == self.FEATURE else "label"


class BaseFeature:

    def show(self):
        raise NotImplemented()


class Feature(BaseFeature):
    def __init__(self, name: str, feat_input: str, factorize: bool,
                 type: ExtractionType, pipeline: ExtractionPipeline,
                 dropped: bool):
        self.name = name
        self.type = type
        self.input = feat_input
        self.factorize = factorize
        self.pipeline = pipeline
        self.dropped = dropped

    def show(self):
        pipeline_repr = " -> ".join(map(repr, self.pipeline.processors))
        print("Standalone Feature (input %s): \n  %s -> %s"
              % (self.input, pipeline_repr, self.name))


class FeatureTree(BaseFeature):
    def __init__(self, feat_input: str, proc_tree: ProcessorsTree,
                 features: List[Feature]):
        self.input = feat_input
        self.proc_tree = proc_tree
        self.features = features

    def show(self):
        print("Feature Tree (input %s):" % self.input)
        self.proc_tree.show()


class Extractor:
    def __init__(self, fail_on_error=True, factorize_features=True):
        self.extractions: Dict[str, Feature] = OrderedDict()
        self.extraction_order: List[BaseFeature] = None
        self.needs_scheduling = False
        self.fail_on_error = fail_on_error
        self.factorize = factorize_features

    def add_extraction(
            self,
            extractors: Union[List[BaseProcessor], BaseProcessor],
            fail_on_error: bool = None,
            drop_on_save: bool = False):
        """
        Adds a single feature extraction pipeline to the extractor.

        :param feature_name: name of the extracted feature, should be unique
        :param feat_input: the input data "fed" to the extraction pipeline
        :param extractors: a list of extractors
        :param fail_on_error: if False, errors are silenced and the sample's
        value for that feature will be None
        :param drop_on_save: do not dump feature to output file
        """
        self.needs_scheduling = True

        # if the fail_on_error parameter is None, defaults to the extractor's
        # fail_on_error
        if fail_on_error is None:
            fail_on_error = self.fail_on_error

        if isinstance(extractors, BaseProcessor):
            pipeline = ExtractionPipeline([extractors], fail_on_error)
        elif isinstance(extractors, list):
            pipeline = ExtractionPipeline(extractors, fail_on_error)
        else:
            raise ValueError("Invalid type for extractors")

        if extraction_type not in ("feature", "label"):
            raise ValueError("Extraction type must be 'feature' or 'label'")
        else:
            extraction_type = (ExtractionType.FEATURE
                               if extraction_type == "feature" else
                               ExtractionType.LABEL)

        pipeline.check_typing()
        self.extractions[feature_name] = Feature(
            feature_name, feat_input, factorize, extraction_type, pipeline,
            drop_on_save)

    def _build_extraction_trees(self):
        # TODO : simplfy into a single extraction tree
        # building extraction trees
        features_groups = OrderedDict()
        self.extraction_order = list()
        for feat_name, feature in self.extractions.items():
            if not feature.factorize:
                self.extraction_order.append(feature)
            else:
                first_proc = feature.pipeline.processors[0]
                features_groups\
                    .setdefault((first_proc, feature.input), [])\
                    .append(feature)

        for (_, feat_input), features in features_groups.items():
            if len(features) == 1:  # no need for a feature tree
                self.extraction_order.append(features[0])
                continue

            proc_tree = ProcessorsTree(self.fail_on_error)
            for feature in features:
                proc_tree.add_pipeline(feature.pipeline.processors,
                                       feature.name)
            feat_tree = FeatureTree(feat_input, proc_tree, features)
            self.extraction_order.append(feat_tree)

        self.needs_scheduling = False

    def show(self):
        if self.needs_scheduling:
            self._build_extraction_trees()

        for extraction in self.extraction_order:
            extraction.show()

    @staticmethod
    def _samples_iterator(input_name: str, dataset: DatasetLoader):
        for sample in dataset:
            yield sample, sample[input_name]

    @staticmethod
    def _run(feature: Feature, iterator, dataset: DatasetLoader):

        try:
            for sample, sample_feat in tqdm(iterator, total=len(dataset)):
                if sample_feat is None:
                    continue

                if feature.type == ExtractionType.FEATURE:
                    sample.store_feature(feature.name, sample_feat,
                                         feature.dropped)
                else:  # label
                    sample.store_label(feature.name, sample_feat,
                                       feature.dropped)
        except Exception as e:
            tb = sys.exc_info()[2]
            raise type(e)(("Error during processing of feature %s : "
                           % feature.name) + str(e)
                          ).with_traceback(tb)

    def run_extractions(self, dataset: DatasetLoader,
                        features: List[str] = None):
        """

        :param dataset: A dataset from which to get the processed samples, and
        in which the extracted features are store
        :param features: If not None, only the listed features are to be
        extracted.
        """
        # TODO : add a feature_wise and sample_wise option to select if the extraction
        #  should be done "along" the samples or the features
        if self.needs_scheduling:
            self._build_extraction_trees()

        for extraction in self.extraction_order:
            if isinstance(extraction, FeatureTree):
                sample_it = self._samples_iterator(extraction.input, dataset)
                extraction.proc_tree.set_sample_iter(sample_it)
                for feat_name in extraction.proc_tree.get_feature_order():
                    if features is not None and feat_name not in features:
                        continue

                    feature = self.extractions[feat_name]
                    print("Running pipeline for %s %s" % (str(feature.type),
                                                          feat_name))
                    self._run(feature, extraction.proc_tree(feat_name),
                              dataset)

            elif isinstance(extraction, Feature):
                if features is not None and extraction.name not in features:
                    continue

                print("Running pipeline for %s %s" % (str(extraction.type),
                                                      extraction.name))
                sample_it = self._samples_iterator(extraction.input, dataset)
                self._run(extraction, extraction.pipeline(sample_it), dataset)
