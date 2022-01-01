from collections import Counter
from io import BytesIO
from itertools import product
from typing import TYPE_CHECKING, List, Union, Any

from .utils import format_annotation

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
    from .extraction_graph import InputNode, FeatureNode, SampleProcessorNode, RootNode

    NODE_COLORS_MAPPING = {
        InputNode: "white",
        FeatureNode: "black",
        SampleProcessorNode: "grey",
        RootNode: "red"
    }

    if isinstance(dag, ExtractionPipeline):
        all_nodes = prepare_pipeline_nodes(dag)
    else:
        all_nodes = prepare_dag_nodes(dag)

    # creating a graph, and adding all nodes to the graph, using their depth as
    # a layer
    pl_graph = nx.DiGraph()
    for node in all_nodes:
        pl_graph.add_node(node.ancestor_hash(), layer=node.depth, label=str(node))

    # adding edges
    for node in all_nodes:
        output_type = None
        if isinstance(node, SampleProcessorNode):
            output_type = node.processor.output_type
        elif isinstance(node, RootNode):
            output_type = "Sample"
        if output_type is Any:
            output_type = None
        pl_graph.add_edges_from(product([node.ancestor_hash()],
                                        [child.ancestor_hash() for child in node.children]),
                                output_type=output_type)

    # rendering graph layout
    graph_layout = nx.multipartite_layout(pl_graph, subset_key="layer")

    # building labels and labels repositioning
    label_dict = {node.ancestor_hash(): str(node) for node in all_nodes}
    labels_layout = {}

    for k, v in graph_layout.items():
        if v[1] > 0:
            labels_layout[k] = (v[0], v[1] + 0.1)
        else:
            labels_layout[k] = (v[0], v[1] - 0.1)

    # finding maximum width of DAG using maximum number of nodes per layer
    dag_width = max(Counter(node.depth for node in all_nodes))
    dag_depth = max(node.depth for node in all_nodes)

    # building node colors
    node_colors = [NODE_COLORS_MAPPING[node.__class__] for node in all_nodes]

    # edges labels list
    edges_labels = {edge: format_annotation(pl_graph.get_edge_data(*edge)["output_type"])
                    for edge in pl_graph.edges
                    if pl_graph.get_edge_data(*edge)["output_type"] is not None}

    # generating plot
    plt.figure(figsize=(dag_depth * 2, dag_width))
    nx.draw_networkx_nodes(pl_graph, graph_layout, node_size=600, node_color=node_colors,
                           edgecolors="black")
    nx.draw_networkx_labels(pl_graph, labels_layout, label_dict)
    nx.draw_networkx_edges(pl_graph, graph_layout, connectionstyle='arc3,rad=-0.2')
    nx.draw_networkx_edge_labels(pl_graph, graph_layout,
                                 edge_labels=edges_labels,
                                 font_size=8,
                                 bbox=dict(alpha=0),
                                 rotate=True)
    plt.axis("equal")
    if show:
        plt.show()

    with BytesIO() as buffer:
        plt.savefig(buffer, format="png")
        png_bytes = buffer.getvalue()

    plt.close()
    return png_bytes
