
# General TODO

* check that samples all have unique ID's at extraction time (maybe done in the root node?)
* check that feature names != input data names
* add skip_errors logic in extraction graph 
  -> decide whether to define it at definition time or extraction time
  -> when an error happens in a processor and skip_error=True, set the sample as None (instead of the processed data)
* decide whether to use Feat("myfeat") or Input("myfeat") for input data from another feature (PROBABLY Feat) (either are ok?)
* check for default args (none should be present ) in Processor.process signature (same with wrapped functions)
* replace asserts with real useful errors
* add support for processors that return a Tuple, and add a specific split symbol
  to support splitting the tuple onto several processors ("tuple output unpacking")
* add support for fully uncached extraction
* for pickle (and maybe hdf5), add support for "direct store" feature (not stored in memory once computed,  
  directly put on disk in the resulting pickle)
* add dataset-level pipelines to compute feature aggregates
* idea for an eventual CLI tool: specify the object to load from a script in the current python namespace.
* use networkX and multipartite_graph to plot the processing DAG
* use extras_requires( `pip install adfluo[plot]`) to install extra plotting dependencies
* documentation on documentation https://diataxis.fr/
* Deactivate settattr on processors (make object frozen) during process call 
* rework the error reporting system (when using skip errors or not)
* DONE : maybe use typevars with param() to prevent having to annotate the parameter -> useless
* make sure that `add_extraction(Input("test"))` works to get a feature directly from an input
* add optional validation logic, either through a `validates` method in sampleprocessor 
  or via a dedicated `SampleValidator` processor.
* URGENT : make feature extraction order cache-efficient (using a tree iterator)
* Use https://github.com/bdcht/grandalf to layout graph and https://github.com/cduck/drawsvg to draw the processor graph
* Make a recipe for resampling (maybe also think about some helpful API elements for this)
* EASY: add "reset" (clear cache and all) functionality to be able to reuse the same extractor on different datasets in the same run
* URGENT: add 'append/overwrite mode' to storages
* URGENT: default to pretty printing data when no format is specified (dict storage).
  -> Should be able to show table if asked as well
* URGENT: check if "possibility of calling a pipeline right away on a sample/dataset" is working
* URGENT: add dataset features
  - Figure out storability
* TODO: use generics for processor type in graph nodes classes
* EASY: use fnmatch (https://docs.python.org/3/library/fnmatch.html#fnmatch.filter) 
  when using features restrictions or inclusions in CLI
* EASY: variable-length inputs using `inspect.signature(f).parameters["args"].kind` on `process` method
* EASY: check that the BadSampleException mechanism works properly (in tests)
* EASY: create NullCache that stores nothing to simplify "if cache" logic?
* MAYBE: use __get__ on hparams classes to simplify things a bit (maybe?)
* EASY: add pre-process hook method for processors
* URGENT : Subset should be on outputs, _not_ inputs
* EASY (?): Use rich's console to redirect logger during extraction https://rich.readthedocs.io/en/stable/progress.html#print-log
* EASY : have two mode for progress: rich & TQDM

# Future implementation Notes

* Add a validator class that can validate inputs from the dataset:

```python

class MyValidator(BaseValidator):
  
  @validates("input_a")
  def validate_a(self, data: TypeA):
    # check that data is valid and return true if it is
    ...
  
  @validates("input_b")
  def validate_b(self, data: TypeB):
    # same for b
    ...
  
  ...

```
- The validator class is then passed to the extractor at instanciation time.
It then will decorate the __iter__ class from the datasetloader, which will 
in turn decorate samples' __getitem__ class.
- OR: Validator nodes are inserted in the graph before/after/in the input nodes
- OR: Validator callbacks are passed to the corresponding input processors
