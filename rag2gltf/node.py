from typing import Optional, List

import glm  # type: ignore

from parsing.rsm import Rsm
from utils import decode_string


class Node:
    def __init__(
        self,
        rsm_node: Rsm.Node,
        children: List['Node'] = [],
        parent: Optional['Node'] = None,
    ):
        self.impl = rsm_node
        self.parent = parent
        self.children = children
        self.bbox = None
        self.local_transform_matrix = glm.mat4()
        self.final_transform_matrix = glm.mat4()
        self.gltf_transform_matrix = glm.mat4()


def extract_nodes(rsm_obj: Rsm) -> List[Node]:
    node_list = []
    for rsm_node in rsm_obj.nodes:
        node_list.append(Node(rsm_node))

    for node in node_list:
        node.children = _find_children_nodes(node_list, node)
        node.parent = _find_parent_node(node_list, node)

    return node_list


def _find_parent_node(nodes: List[Node], node: Node) -> Optional[Node]:
    parent_name = decode_string(node.impl.parent_name)
    if len(parent_name) == 0:
        return None
    for other_node in nodes:
        if other_node != node and decode_string(
                other_node.impl.name) == parent_name:
            return other_node
    return None


def _find_children_nodes(nodes: List[Node], node: Node) -> List[Node]:
    children = []
    node_name = decode_string(node.impl.name)
    for other_node in nodes:
        parent_name = decode_string(other_node.impl.parent_name)
        if other_node == node or len(parent_name) == 0:
            continue
        if parent_name == node_name:
            children.append(other_node)
    return children