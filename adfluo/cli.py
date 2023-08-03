import json
import logging
import os
import sys
from argparse import ArgumentParser
from collections import Counter
from importlib import import_module
from pathlib import Path
from typing import Optional, List, Dict, Union

from tqdm import tqdm
from typing_extensions import Literal

from adfluo import DatasetLoader, Extractor, Sample
from adfluo.dataset import ListLoader, SubsetLoader
from .utils import logger, extraction_policy


class CLIParametersError(Exception):
    pass


def import_class(class_path: str) -> Optional[Union[Extractor, DatasetLoader, List[Dict]]]:
    # TODO: better error
    assert len(class_path.split(".")) > 1
    *module_path, obj_name = class_path.split(".")
    module_path = ".".join(module_path)

    sys.path.append(os.getcwd())
    try:
        mod = import_module(module_path)
        obj = getattr(mod, obj_name)
    except (ImportError, AttributeError) as err:
        return None
    else:
        return obj


# TODO: add support for dataset_args
def load_dataset(dataset_name: str) -> DatasetLoader:
    # first trying to load from json (dataset is a path)
    dataset_path = Path(dataset_name)
    if dataset_path.is_file() and dataset_path.suffix == ".json":
        with open(dataset_path) as json_file:
            json_data = json.load(json_file)
            return ListLoader(json_data)

    else:
        obj = import_class(dataset_name)
        if isinstance(obj, list):
            return ListLoader(obj)
        elif isinstance(obj, DatasetLoader):
            return obj
        elif obj is None:
            raise CLIParametersError(f"Couldn't import any dataset with name {dataset_name}")
        else:
            raise CLIParametersError(f"{dataset_name} isn't a valid dataset object")


class Command:
    COMMAND = "command"
    DESCRIPTION = "Command description"

    @staticmethod
    def init_parser(parser: ArgumentParser):
        pass

    @classmethod
    def main(cls, **kwargs):
        pass


class ExtractCommand(Command):
    COMMAND = "extract"
    DESCRIPTION = "Command description"

    @staticmethod
    def init_parser(parser: ArgumentParser):
        parser.add_argument("extractor", type=str,
                            help="Extractor instance in the current namespace")
        parser.add_argument("dataset", type=str,
                            help="Either a path to a json file "
                                 "that has a dataset layout, a list of samples, "
                                 "a DatasetLoader instance, or a DatasetLoader "
                                 "subclass")
        parser.add_argument("--dataset_args", type=str, nargs="*",
                            help="If the dataset argument is a class, "
                                 "these are passed as the class's "
                                 "instantiation parameters")
        action = parser.add_mutually_exclusive_group(required=True)
        action.add_argument("--output", "-o", type=Path,
                            help="Output file path or folder (depending on format)")
        action.add_argument("--test_samples",
                            action="store_true",
                            help="Just test that samples all can all provide "
                                 "the required input data")
        parser.add_argument("--feats", "-f", nargs="*", type=str,
                            help="Extract only for the specified features")
        parser.add_argument("--samples", "-s", nargs="*", type=str,
                            help="Extract only for the specified samples")
        parser.add_argument("--indexing",
                            choices=["feature", "sample"],
                            default="sample",
                            help="Storage indexing policy")
        parser.add_argument("--order",
                            choices=["feature", "sample"],
                            default="feature",
                            help="Extraction order (feature-wise or sample-wise)")
        parser.add_argument("--skip_errors",
                            action="store_true",
                            help="Errors while computing a feature for a sample are ignored.")
        parser.add_argument("--no_caching",
                            action="store_true",
                            help="Disable any form of caching (may impact performances "
                                 "but prevents memory overflows")
        parser.add_argument("--storage_format", "-sf", type=str,
                            choices=["csv", "json", "df", "pickle", "split-pickle", "hdf5"],
                            default="pickle",
                            help="Storage format for the extracted features")

        parser.add_argument("--hide_progress",
                            action="store_true",
                            help="Don't show progress bars during the extraction")

    @classmethod
    def main(cls,
             extractor: str,
             dataset: str,
             dataset_args: List[str],
             output: Path,
             feats: List[str],
             samples: List[str],
             indexing: Literal["feature", "sample"],
             order: Literal["feature", "sample"],
             skip_errors: bool,
             no_caching: bool,
             storage_format: Literal["csv", "json", "df", "pickle", "split-pickle", "hdf5"],
             test_samples: bool,
             hide_progress: bool,
             **kwargs):

        extractor: Extractor = import_class(extractor)
        if not isinstance(extractor, Extractor):
            raise CLIParametersError(f"{extractor} isn't an extractor object")
        elif extractor is None:
            raise CLIParametersError(f"Couldn't import extractor {extractor}")

        dataset: DatasetLoader = load_dataset(dataset)

        if test_samples:
            error_count = 0
            for sample in tqdm(dataset):
                sample: Sample
                for input_name in extractor.extraction_DAG.inputs:
                    try:
                        sample[input_name]
                    except Exception as err:
                        print(f"Got error '{type(err)} : {err}' on sample {sample.id} for "
                              f"when asked to provide input '{input_name}'")
                        error_count += 1

            print(f"Got {error_count} errors when testing {len(dataset)} samples from {dataset}")
            exit()

        # keeping only features that are specified in `feats`
        feats = set(feats)
        if feats:
            extractor.extraction_DAG.feature_nodes = {feat: feat_node for feat, feat_node
                                                      in extractor.extraction_DAG.feature_nodes.items()
                                                      if feat in feats}

        # wrapping the dataset with a subsetloader if only a subset of samples has been specified
        if samples:
            dataset = SubsetLoader(dataset, samples)

        if storage_format == "csv":
            extraction_method = extractor.extract_to_csv
        elif storage_format == "json":
            extraction_method = extractor.extract_to_json
        elif storage_format == "pickle":
            extraction_method = extractor.extract_to_pickle
        elif storage_format == "split-pickle":
            extraction_method = extractor.extract_to_pickle_files
        elif storage_format == "hdf5":
            extraction_method = extractor.extract_to_hdf5
        else:
            raise ValueError("Invalid extraction format")

        kwargs = {
            "extraction_order": order,
            "storage_indexing": indexing,
            "no_caching": no_caching
        }
        if storage_format == "split-pickle":
            kwargs["output_folder"] = output
        else:
            kwargs["output_file"] = output

        extraction_policy.skip_errors = skip_errors
        extractor.show_progress = not hide_progress

        extraction_method(dataset, **kwargs)


class ShowCommand(Command):
    COMMAND = "show"
    DESCRIPTION = "Show informations about an extractor or a dataset"

    @staticmethod
    def init_parser(parser: ArgumentParser):
        # TODO: add dataset args
        parser.add_argument("extractor_or_dataloader", type=str,
                            help="Either a DataLoader or Extractor instance in the current Namespace")
        parser.add_argument("--output_file", "-o", type=Path,
                            help="Output file path for the extraction DAG's plot")
        # TODO : option to show DAG tree if possible
        parser.add_argument("--dag", action="store_true",
                            help="If we're dealing ")

    @classmethod
    def main(cls,
             extractor_or_dataloader: str,
             output_file: Optional[Path],
             dag: bool,
             **kwargs):
        obj = import_class(extractor_or_dataloader)
        if obj is None:
            logger.error(f"Couldn't import extractor or dataset class {extractor_or_dataloader}")
            exit(1)

        # TODO: attempt at loading from json

        if isinstance(obj, list):
            obj = ListLoader(obj)

        if isinstance(obj, Extractor):
            print(f"Info for extractor {extractor_or_dataloader}")

            print(f"{len(obj.extraction_DAG.inputs)} inputs required:")
            for input_name in obj.extraction_DAG.inputs:
                print(f"\t- {input_name}")

            print(f"{len(obj.extraction_DAG.features)} features specified:")
            for feat_name in obj.extraction_DAG.features:
                print(f"\t- {feat_name}")

            # TODO: if dag output is specified, print to PNG/else

        elif isinstance(obj, DatasetLoader):
            print(f"Info for dataset {extractor_or_dataloader}:")
            print(f"{len(obj)} samples")
            sample_ids = [sample.id for sample in obj]
            samples_counts = Counter(sample_ids)
            duplicates = sorted([sample_id
                                 for sample_id, count in samples_counts.items()
                                 if count > 1])
            if duplicates:
                print(f"WARNING: The following samples ids are duplicate: "
                      f"{', '.join(duplicates)}")


commands = [ExtractCommand,
            ShowCommand]

argparser = ArgumentParser()
argparser.add_argument("-v", "--verbose",
                       action="store_true",
                       help="Verbose mode")
subparsers = argparser.add_subparsers()
for command in commands:
    parser = subparsers.add_parser(command.COMMAND)
    parser.set_defaults(func=command.main)
    command.init_parser(parser)


def main():
    args = argparser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # Invoking the right subfunction
    try:
        # calling the right command with Namespace converted to kwargs
        args.func(**vars(args))
    except CLIParametersError as err:
        logger.error(str(err))
        exit(1)


if __name__ == '__main__':
    main()
