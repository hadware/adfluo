
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
* ask about names for 
  - the extractor (ExtractionSet?)
  - DatasetLoader?
* possibility of calling a pipeline right away on a sample/dataset
* documentation on documentation https://diataxis.fr/
* Deactivate settattr (make object frozen) during process call 
* rework the error reporting system (when using skip errors or not)
* maybe use typevars with param() to prevent having to annotate the parameter
* make sure that `add_extraction(Input("test"))` works to get a feature directly from an input
* add optional validation logic, either through a `validates` method in sampleprocessor 
  or via a dedicated `SampleValidator` processor.
* URGENT : make feature extraction order cache-efficient (using a tree iterator)
* URGENT : rename "pickle per file" to "split-pickle"
* URGENT?: add support for automatic list-of-items processing via *ProcessorInstance() (overloading __iter__ to return a wrapped processor)
* Use PyCairo to draw the processor graph
* Make a recipe for resampling (maybe also think about some helpful API elements for this)
* EASY: add custom storage that can be passed to `Feat` (callable with signature (feat_name, sample_id, data: Any))
* EASY: use rich.track instead of tqdm: it's much prettier. track also has a `disable` setting
* EASY: add "reset" (clear cache and all) functionality to be able to reuse the same extractor on different datasets in the same run
* EASY - URGENT: Add a check() function to storage backends that checks if each and every value can be safely dumped

# Future implementation Notes

* Regarding the factorization of pipeline DAG's: 
  - first merge the pipeline in the main DAG, considering all feature inputs as
    regular inputs
  - then run a pass to check which inputs are feature and "connect"
    these to their feature "leaves". This is **necessary** if we don't want 
    to have to run a dependency algorithm before running the extraction. In this
    case, dependencies between features are "naturally" expressed through the tree.
* Regarding the CLI tool (feature ideas, at random):
  - for an extraction set, required inputs and the features it extracts
  - display DAG (to PNG or as a matplotlib window)
  - run an extraction
    - select one or more features
    - select the savemode (CSV, DF, Pickle, HDF5)
    - just test if the required samples load
  - for a dataset: the sample count
  
```shell
adfluo extract module.my_extractor module.my_dataloader --feats f_a f_b --samples samp_a samp_b
adfluo extract module.my_extractor module.my_dataloader --format csv
adfluo extract module.my_extractor module.my_dataloader --format pickle -o filename.pckl
adfluo extract module.my_extractor module.my_dataloader --test_samples # test all required data fields
adfluo extract module.my_extractor module.my_dataloader --indexing feature
adfluo extract module.my_extractor module.my_dataloader --extraction_order sample
adfluo extract module.my_extractor module.my_dataloader --hide_progress
adfluo show module.my_extractor
adfluo show module.my_extractor --dag -o dag.png
adfluo show module.my_dataloader
```

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
