from collections import Counter
from io import BytesIO
from itertools import product
from typing import TYPE_CHECKING, List, Union

if TYPE_CHECKING:
    from .pipeline import ExtractionPipeline
    from .extraction_graph import ExtractionDAG, BaseGraphNode
    from networkx import DiGraph


def prepare_pipeline_nodes(pl: 'ExtractionPipeline') -> List['BaseGraphNode']:
    # reseting node depths to prevent fuckups
    for node in pl.all_nodes:
        node.depth = None

    # setting the input nodes of a pipeline as a depth of 0
    for node in pl.inputs:
        node.depth = 0

    return pl.all_nodes


def prepare_dag_nodes(dag: 'ExtractionDAG') -> List['BaseGraphNode']:
    # reseting node depths to prevent fuckups
    for node in dag.nodes:
        node.depth = None

    # setting root_node depth to 0
    dag.root_node.depth = 0

    return [dag.root_node] + dag.nodes


def plot_dag(dag: Union['ExtractionPipeline', 'ExtractionDAG'],
             show: bool = False) -> bytes:
    try:
        import networkx as nx
    except ImportError:
        raise ImportError(
            "Cannot plot without the networkx package. Please run `pip install adfluo[plot]`")

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError(
            "Cannot plot without the matplotlib package. Please run `pip install adfluo[plot]`"
        )

    from .pipeline import ExtractionPipeline

    if isinstance(dag, ExtractionPipeline):
        all_nodes = prepare_pipeline_nodes(dag)
    else:
        all_nodes = prepare_dag_nodes(dag)

    # creating a graph, and adding all nodes to the graph, using their depth as
    # a layer
    pl_graph = nx.DiGraph()
    nodes_ids = {node: node.ancestor_hash() for node in all_nodes}
    for node, node_id in nodes_ids.items():
        pl_graph.add_node(node_id, layer=node.depth)

    # adding edges
    for node, node_id in nodes_ids.items():
        pl_graph.add_edges_from(product([node_id],
                                        [nodes_ids[child] for child in node.children]))

    # rendering graph layout
    graph_layout = nx.multipartite_layout(pl_graph, subset_key="layer")

    # building labels and labels repositioning
    label_dict = {nodes_ids[node]: str(node) for node in all_nodes}
    labels_layout = {}

    for k, v in graph_layout.items():
        if v[1] > 0:
            labels_layout[k] = (v[0], v[1] + 0.1)
        else:
            labels_layout[k] = (v[0], v[1] - 0.1)

    # finding maximum width of DAG using maximum number of nodes per layer
    dag_width = max(Counter(node.depth for node in all_nodes))
    dag_depth = max(node.depth for node in all_nodes)

    # generating plot
    plt.figure(figsize=(dag_depth * 2, dag_width))
    nx.draw(pl_graph, graph_layout,
            node_size=400,
            font_size=10)
    nx.draw_networkx_labels(pl_graph, labels_layout, label_dict)
    plt.axis("equal")
    if show:
        plt.show()

    with BytesIO() as buffer:
        plt.savefig(buffer, format="png")
        png_bytes = buffer.getvalue()

    plt.close()
    return png_bytes
