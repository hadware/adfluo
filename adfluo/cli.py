import logging
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Optional, List, Dict

from adfluo import DatasetLoader, Extractor
from adfluo.dataset import ListLoader
from .utils import logger


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


def extract(args: Namespace):
    pass


def show(args: Namespace):
    # TODO: show duplicate id's in the dataset report
    obj = import_class(args.extractor_or_dataloader)
    if obj is None:
        logging.error(f"Couldn't import extractor or dataset class {args.extractor_or_dataloader}")
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
    parser_extract.add_argument("--feats", nargs="*", type=str,
                                help="Extract only for the specified features")
    parser_extract.add_argument("--samples", nargs="*", type=str,
                                help="Extract only for the specified samples")
    parser_extract.add_argument("--indexing",
                                choices=["feature", "sample"],
                                default="sample",
                                help="Storage indexing policy")
    parser_extract.add_argument("--extraction_order",
                                choices=["feature", "sample"],
                                default="feature",
                                help="Extraction order (feature-wise or sample-wise")
    parser_extract.add_argument("--format", "-f", type=str,
                                choices=["csv", "df", "pickle", "hdf5"],
                                default="pickle",
                                help="Storage format for the extracted features")
    parser_extract.add_argument("--output_file", "-o", type=Path,
                                help="Output file path")
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
    args.func(args)
