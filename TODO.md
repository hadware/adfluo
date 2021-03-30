
# General TODO

* check that samples all have unique ID's at extraction time (maybe done in the root node?)
* check that feature names != input data names
* change fail_on_error to skip_errors
* add skip_errors logic in extraction graph 
  -> decide whether to define it at definition time or extraction time
  -> when an error happens in a processor and skip_error=True, set the sample as None (instead of the processed data)
* decide whether to use Feat("myfeat") or Input("myfeat") for input data from another feature (PROBABLY Feat) (either are ok?)
* check for default args (none should be present ) in Processor.process signature (same with wrapped functions)
* replace asserts with real useful errors
* add support for processors that return a Tuple, and add a specific split symbol
  to support splitting the tuple onto several processors ("tuple output unpacking")
* add support for list of samples/ and samples-as-dicts for input datasets
* add support for csv, df, pickle and dict extraction
* add support for per feature/sample pickle file (for very large features) -> in a new method
* add support for drop-on-save features (maybe find a better name)
* add support for fully uncached extraction
* for pickle (and maybe hdf5), add support for "direct store" feature (not stored in memory once computed,  
  directly put on disk in the resulting pickle)
* add dataset-level pipelines to compute feature aggregates
* for processor params, add a dataclass-aware system that uses the dataclass
  attributes as parameters
* idea for an eventual CLI tool: specify the object to load from a script in the current python namespace.
* use networkX and multipartite_graph to plot the processing DAG
* use extras_requires( `pip install adfluo[plot]`) to install extra plotting dependencies
* ask about names for 
  - the extractor (ExtractionSet?)
  - DatasetLoader?

# Future implementation Notes

* Regarding the factorization of pipeline DAG's: 
  - first merge the pipeline in the main DAG, considering all feature inputs as
    regular inputs
  - then run a pass to check which inputs are feature and "connect"
    these to their feature "leaves". This is **necessary** if we don't want 
    to have to run a dependency algorithm before running the extraction. In this
    case, dependencies between features are "naturally" expressed through the tree.
  - in this case, features would then become a passthrough processor
  - the cache mecanism could be tricky (the data could be stored twice: once in the
    feature's DAG node, and once in the sample's feature storage
* Regarding the CLI tool (feature ideas, at random):
  - for an extraction set, required inputs and the features it extracts
  - display DAG (to PNG or as a matplotlib window)
  - run an extraction
    - select one or more features
    - select the savemode (CSV, DF, Pickle, HDF5)
    - just test if the required samples load
  - for a dataset: the sample count