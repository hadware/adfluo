
* check that samples all have unique ID's at extraction time
* check that feature names != input data names
* add fail_on_error logic in extraction graph
* modify the processor class to work without iterators
* think about/add combiner processors
    - as regular processors with several arguments (prefered choice)
    - as a separate class of processors?
* add lambda processor from soranos
* remove type-checking code -> DONE
* add inter-feature dependency graph
* remove input nodes and replace by regular processor nodes
* decide whether or not to use Feature("myfeat") or Input("myfeat") for input
  data from another feature (PROBABLY YES?)