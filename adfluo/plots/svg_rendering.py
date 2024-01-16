import io
import math
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Literal, Union
from typing import Tuple, List, Optional

from grandalf.graphs import Vertex, Edge, Graph
from grandalf.routing import EdgeViewer, route_with_lines

from .sphinx_utils import get_type_hints, stringify_annotation
from ..extraction_graph import BaseGraphNode, InputNode, SampleProcessorNode, RootNode, FeatureNode, DatasetFeatureNode, \
    AggregatorNode, DatasetInputNode

if TYPE_CHECKING:
    from ..extraction_graph import ExtractionDAG
    from ..pipeline import ExtractionPipeline
    from grandalf.layouts import SugiyamaLayout
    from grandalf.graphs import Graph
    from drawsvg import Drawing

Point = Tuple[float, float]


def rotate(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = round(x * math.cos(radians) + y * math.sin(radians), 0)
    yy = round(-x * math.sin(radians) + y * math.cos(radians), 0)
    return xx, yy


def svg_coord(point: Point) -> Point:
    return point[0], abs(point[1])


@dataclass
class GraphNode:
    radius: float
    center: Point
    ref: BaseGraphNode

    def top_left(self) -> tuple[float, float]:
        return self.center[0] - self.radius / 2, self.center[1] + self.radius / 2

    def bottom_right(self) -> tuple[float, float]:
        return self.center[0] + self.radius / 2, self.center[1] - self.radius / 2

    def translate(self, offset: Point):
        self.center = (self.center[0] + offset[0], self.center[1] + offset[1])

    def rotate(self, radians: float):
        self.center = rotate(self.center, radians)


@dataclass
class GraphEdge:
    points: List[Point]
    head_angle: float
    start_node: BaseGraphNode
    end_node: BaseGraphNode

    def translate(self, offset: Point):
        self.points = [(point[0] + offset[0], point[1] + offset[1]) for point in self.points]

    def rotate(self, radians: float):
        self.head_angle -= radians
        self.points = [rotate(point, radians) for point in self.points]

    def start_output_sig(self) -> Optional[str]:
        if not isinstance(self.start_node, (SampleProcessorNode, AggregatorNode)):
            return None

        if isinstance(self.start_node, SampleProcessorNode):
            proc_fn = self.start_node.processor.process
        else:
            proc_fn = self.start_node.processor.aggregate

        sig = get_type_hints(proc_fn)
        if "return" in sig:
            return stringify_annotation(sig["return"])
        else:
            return None


class SVGGraphRenderer:
    NODE_RADIUS: int = 40
    NODE_COLORS_MAPPING = {
        InputNode: "white",
        DatasetInputNode: "white",
        FeatureNode: "blue",
        DatasetFeatureNode: "blue",
        SampleProcessorNode: "grey",
        AggregatorNode: "grey",
        RootNode: "red"
    }

    def build_layout(self, nodes: List[BaseGraphNode]) -> Tuple['Graph', 'SugiyamaLayout']:
        vertices_dict = {node: Vertex(data=node) for node in nodes}
        edges = []
        for node in nodes:
            for start_node, end_node in product([node], node.children):
                edges.append(Edge(vertices_dict[start_node], vertices_dict[end_node]))
        graph = Graph(list(vertices_dict.values()), edges)

        class defaultview:
            w, h = self.NODE_RADIUS, self.NODE_RADIUS

        for v in vertices_dict.values():
            v.view = defaultview()

        sug = SugiyamaLayout(graph.C[0])
        sug.yspace = self.NODE_RADIUS * 5
        sug.xspace = self.NODE_RADIUS * 2
        sug.init_all()
        sug.draw(3)
        for e in graph.E():
            e.view = EdgeViewer()
        sug.route_edge = route_with_lines
        sug.draw_edges()
        return graph, sug

    def convert_layout(self, graph: 'Graph', layout: 'SugiyamaLayout') -> Tuple[List[GraphNode], List[GraphEdge]]:
        graph_nodes = [GraphNode(v.view.h, v.view.xy, v.data)
                       for v in graph.C[0].sV]
        graph_edges = [GraphEdge(e.view._pts, e.view.head_angle, e.v[0].data, e.v[1].data)
                       for e in layout.g.E()]
        for element in graph_nodes + graph_edges:
            element.rotate(math.radians(90))
        x_min = min([n.top_left()[0] for n in graph_nodes])
        y_max = max([n.top_left()[1] for n in graph_nodes])
        offset = (-(x_min - self.NODE_RADIUS * 2), -(y_max + self.NODE_RADIUS * 2))
        for element in graph_nodes + graph_edges:
            element.translate(offset)
        return graph_nodes, graph_edges

    def build_drawing(self, nodes: List[GraphNode], edges: List[GraphEdge]) -> 'Drawing':
        import drawsvg as draw

        x_max = max([n.bottom_right()[0] for n in nodes])
        y_min = min([n.bottom_right()[1] for n in nodes])

        d = draw.Drawing(x_max + self.NODE_RADIUS, abs(y_min) + self.NODE_RADIUS, origin=(0, 0))
        g = draw.Group(stroke='black', fill='none')
        arrow = draw.Lines(0, - self.NODE_RADIUS / 7,
                           self.NODE_RADIUS / 2, 0,
                           0, self.NODE_RADIUS / 7,
                           0, - self.NODE_RADIUS / 7,
                           fill="black")
        for node in nodes:
            center_coords = svg_coord(node.center)
            g.append(draw.Circle(*center_coords, node.radius * 0.8,
                                 stroke=self.NODE_COLORS_MAPPING[node.ref.__class__],
                                 stroke_width=5))
            g.append(draw.Text(str(node.ref), 30,
                               center_coords[0], center_coords[1] + (node.radius * 1.3),
                               text_anchor="middle", dominant_baseline="middle"))
        for edge in edges:
            edge_path = draw.Path(stroke_width=3).M(*svg_coord(edge.points[0]))
            for point in edge.points[1:]:
                edge_path = edge_path.T(*svg_coord(point))
            g.append(edge_path)
            start_node_annot = edge.start_output_sig()
            if start_node_annot is not None:
                g.append(draw.Text(start_node_annot, 30, path=edge_path, text_anchor="middle"))
            end_coords = svg_coord(edge.points[-1])
            g.append(draw.Use(arrow, *end_coords,
                              transform=f"rotate ({-math.degrees(edge.head_angle)},{end_coords[0]},{end_coords[1]})"))
        d.append(g)
        return d

    def render(self, dag: Union['ExtractionPipeline', 'ExtractionDAG'],
               img_format: Literal["png", "svg"] = "svg") -> bytes:
        assert img_format in {"png", "svg"}

        from ..extraction_graph import ExtractionDAG
        from ..pipeline import ExtractionPipeline
        if isinstance(dag, ExtractionPipeline):
            all_nodes = dag.all_nodes
        elif isinstance(dag, ExtractionDAG):
            all_nodes = list(dag.nodes | {dag.root_node})
        else:
            raise TypeError("Unsupported object type for dag")

        graph, layout = self.build_layout(all_nodes)
        nodes, edges = self.convert_layout(graph, layout)
        drawing = self.build_drawing(nodes, edges)

        if img_format == "svg":
            return drawing.as_svg()
        else:
            bytes_io = io.BytesIO()
            drawing.save_png(fname=bytes_io)
            return bytes_io.getvalue()
