import json
import logging
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict

from adfluo import DatasetLoader, Extractor
from adfluo.dataset import ListLoader
from .utils import logger, extraction_policy


class CLIParametersError(Exception):
    pass


def import_class(class_path: str) -> Optional[Extractor,
                                              DatasetLoader,
                                              List[Dict]]:
    # TODO: better error
    assert len(class_path.split(".")) > 1
    *module_path, obj_name = class_path.split(".")
    module_path = ".".join(module_path)
    try:
        mod = __import__(module_path, fromlist=[obj_name])
        obj = getattr(mod, obj_name)
    except (ImportError, AttributeError):
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
        obj = import_class(args.extractor_or_dataloader)
        if isinstance(obj, list):
            return ListLoader(obj)
        elif isinstance(obj, DatasetLoader):
            return obj
        elif obj is None:
            raise CLIParametersError(f"Couldn't import any dataset with name {dataset_name}")
        else:
            raise CLIParametersError(f"{dataset_name} isn't a valid dataset object")


def extract(args: Namespace):
    extractor: Extractor = import_class(args.extractor)
    if not isinstance(extractor, Extractor):
        raise CLIParametersError(f"{args.extractor} isn't an extractor object")
    elif extractor is None:
        raise CLIParametersError(f"Couldn't import extractor {args.extractor_or_dataloader}")

    dataset: DatasetLoader = load_dataset(args.dataset)

    if args.test_samples:
        pass  # TODO

    extraction_format: str = args.format
    if extraction_format == "csv":
        extraction_method = extractor.extract_to_csv
    elif extraction_format == "json":
        extraction_method = extractor.extract_to_json
    elif extraction_format == "pickle":
        extraction_method = extractor.extract_to_pickle
    elif extraction_format == "split-pickle":
        extraction_method = extractor.extract_to_pickle_files
    elif extraction_format == "hdf5":
        extraction_method = extractor.extract_to_hdf5
    else:
        raise ValueError("Invalid extraction format")

    kwargs = {
        "extraction_order": args.order,
        "storage_indexing": args.indexing,
        "no_caching": args.no_caching
    }
    if extraction_format == "split-pickle":
        kwargs["output_folder"] = args.output
    else:
        kwargs["output_file"] = args.output

    extraction_policy.skip_errors = args.skip_errors

    extraction_method(dataset, **kwargs)

def show(args: Namespace):
    obj = import_class(args.extractor_or_dataloader)
    if obj is None:
        logger.error(f"Couldn't import extractor or dataset class {args.extractor_or_dataloader}")
        exit(1)

    # TODO: attempt at loading from json

    if isinstance(obj, list):
        obj = ListLoader(obj)

    if isinstance(obj, Extractor):
        print(f"Info for extractor {args.extractor_or_dataloader}")

        print(f"{len(obj.extraction_DAG.inputs)} required:")
        for input_name in obj.extraction_DAG.inputs:
            print(f"\t- {input_name}")

        print(f"{len(obj.extraction_DAG.features)} specified:")
        for feat_name in obj.extraction_DAG.features:
            print(f"\t- {feat_name}")

        # TODO: if dag output is specified, print to PNG/else

    elif isinstance(obj, DatasetLoader):
        print(f"Info for dataset {args.extractor_or_dataloader}:")
        print(f"{len(obj)} samples")
        sample_ids = [sample.id for sample in obj]
        samples_counts = Counter(sample_ids)
        duplicates = sorted([sample_id
                             for sample_id, count in samples_counts.values()
                             if count > 1])
        if duplicates:
            print(f"WARNING: The following samples ids are duplicate: "
                  f"{', '.join(duplicates)}")


if __name__ == '__main__':
    argparser = ArgumentParser("adfluo")
    argparser.add_argument("--verbose", "-v",
                           action="store_true",
                           help="Verbose mode")

    subparsers = argparser.add_subparsers()
    parser_extract = subparsers.add_parser("extract")
    parser_extract.set_defaults(func=extract)
    parser_extract.add_argument("extractor", type=str,
                                help="Extractor instance in the current namespace")
    parser_extract.add_argument("dataset", type=str,
                                help="Either a path to a json file "
                                     "that has a dataset layout, a list of samples, "
                                     "a DatasetLoader instance, or a DatasetLoader "
                                     "subclass")
    parser_extract.add_argument("--dataset_args", type=str, nargs="*",
                                help="If the dataset argument is a class, "
                                     "these are passed as the class's "
                                     "instantiation parameters")
    parser_extract.add_argument("--output", "-o", type=Path, required=True,
                                help="Output file path or folder (depending on format)")
    parser_extract.add_argument("--feats", nargs="*", type=str,
                                help="Extract only for the specified features")
    parser_extract.add_argument("--samples", nargs="*", type=str,
                                help="Extract only for the specified samples")
    parser_extract.add_argument("--indexing",
                                choices=["feature", "sample"],
                                default="sample",
                                help="Storage indexing policy")
    parser_extract.add_argument("--order",
                                choices=["feature", "sample"],
                                default="feature",
                                help="Extraction order (feature-wise or sample-wise")
    parser_extract.add_argument("--skip_errors",
                                action="store_true",
                                help="Errors while computing a feature for a sample "
                                     "are ignored.")
    parser_extract.add_argument("--no_caching",
                                action="store_true",
                                help="Disable any form of caching (may impact performances "
                                     "but prevents memory overflows")
    parser_extract.add_argument("--format", "-f", type=str,
                                choices=["csv", "df", "pickle", "split-pickle", "hdf5"],
                                default="pickle",
                                help="Storage format for the extracted features")
    parser_extract.add_argument("--test_samples",
                                action="store_true",
                                help="Just test that samples all can all provide "
                                     "the required input data")
    parser_extract.add_argument("--hide_progress",
                                action="store_true",
                                help="Don't show progress bars during the extraction")

    parser_show = subparsers.add_parser("show")
    parser_show.set_defaults(func=show)
    # TODO: add dataset args
    parser_show.add_argument("extractor_or_dataloader", type=str,
                             help="Either a DataLoader or Extractor instance in the current Namespace")
    parser_extract.add_argument("--output_file", "-o", type=Path,
                                help="Output file path for the extraction DAG's plot")
    # TODO : option to show DAG tree if possible
    parser_extract.add_argument("--dag", action="store_true",
                                help="If we're dealing ")

    args = argparser.parse_args()
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    # Invoking the right subfunction
    try:
        args.func(args)
    except CLIParametersError as err:
        logger.error(str(err))
        exit(1)
