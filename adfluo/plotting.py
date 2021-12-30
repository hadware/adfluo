from itertools import product
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import ExtractionPipeline
    from .extraction_graph import ExtractionDAG

NETWORKX_IMPORT_ERR = ImportError(
    "Cannot plot without the networkx package. Please run `pip install adfluo[plot]`")

# TODO: write plots to buffer, return the bytes, then use
#  https://stackoverflow.com/questions/64099834/how-can-i-display-an-image-in-python-thats-represented-as-an-array-of-bytes-wit
#  to show them in noteboo

def plot_pipeline(pl: ExtractionPipeline) -> bytes:
    try:
        import networkx as nx
    except ImportError:
        raise NETWORKX_IMPORT_ERR

    # setting the input nodes of a pipeline as a depth of 0from
    for node in pl.inputs:
        node.depth = 0

    # creating a graph, and adding all nodes to the graph, using their depth as
    # a layer
    pl_graph = nx.Graph()
    for node in pl.all_nodes:
        pl_graph.add_node(node, layer=node.depth)

    # adding edges
    for node in pl.all_nodes:
        pl_graph.add_edges_from(product([node], node.children))



def plot_extraction_DAG(graph: ExtractionDAG) -> bytes:
    try:
        import networkx as nx
    except ImportError:
        raise NETWORKX_IMPORT_ERR
    # graph plotting roadmap:
    # - create a graph, precompute the multipartite subsets using
    #   the node depth
    # - use "multipartite_layout" to get plotting positions for the graph
    # look into the difference between the graphviz / matplotlib backends
