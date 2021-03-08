from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import ExtractionPipeline
    from .extraction_graph import ExtractionDAG

def plot_pipeline(pl: ExtractionPipeline):
    try:
        import networkx
    except ImportError:
        pass # TODO: error


def plot_extraction_DAG(graph: ExtractionDAG):
    try:
        import networkx
    except ImportError:
        pass  # TODO: error
    # graph plotting roadmap:
    # - create a graph, precompute the multipartite subsets using
    #   the node depth
    # - use "multipartite_layout" to get plotting positions for the graph
    # look into the difference between the graphviz / matplotlib backends