from argparse import ArgumentParser, Namespace
from pathlib import Path


# Used to import from string:
def import_class(class_path: str):
    # TODO: better error
    assert len(class_path.split(".")) > 1
    *module_path, klass_name = class_path.split(".")
    module_path = ".".join(module_path)
    mod = __import__(module_path, fromlist=[klass_name])
    klass = getattr(mod, klass_name)
    return klass


def extract(args: Namespace):
    pass


def show(args: Namespace):
    pass # TODO: show duplicate id's in the dataset report


if __name__ == '__main__':
    argparser = ArgumentParser("adfluo")
    subparsers = argparser.add_subparsers()

    parser_extract = subparsers.add_parser("extract")
    parser_extract.set_defaults(func=extract)
    parser_extract.add_argument("extractor", type=str,
                                help="Extractor instance in the current namespace")
    parser_extract.add_argument("dataloader", type=str,
                                help="DataLoader instance in the current namespace")
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
    parser_extract.add_argument("--verbose", "-v",
                                action="store_true",
                                help="Verbose mode")

    parser_show = subparsers.add_parser("show")
    parser_show.set_defaults(func=show)
    parser_show.add_argument("extractor_or_dataloader", type=str,
                             help="Either a DataLoader or Extractor instance in the current Namespace")
    parser_extract.add_argument("--output_file", "-o", type=Path,
                                help="Output file path for the extraction DAG's plot")
    parser_extract.add_argument("--dag", action="store_true",
                                help="If we're dealing ")

    args = argparser.parse_args()
    # Invoking the right subfunction
    args.func(args)
