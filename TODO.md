
* check that samples all have unique ID's at extraction time (maybe done in the root node?)
* check that feature names != input data names
* change fail_on_error to skip_errors
* add skip_errors logic in extraction graph
* add inter-feature dependency graph (not needed?)
* decide whether or not to use Feat("myfeat") or Input("myfeat") for input
  data from another feature (PROBABLY Feat) (either are ok?)
* check for default args (none should be present ) in Processor.process signature (same with wrapped functions)
* replace asserts with real useful errors
* add support for processor that return a Tuple, and add a specific split symbol
  to support splitting the tuple onto several processors ("tuple output unpacking")
* add support for list of samples/ and samples-as-dicts for input datasets
* add support for csv, df, pickle and dict extraction
* add support for drop-on-save features (maybe find a better name)
* add support for fully uncached extraction
* for pickle (and maybe hdf5), add support for "direct store" feature (not stored in memory once computed, 
  directly put on disk in the resulting pickle)