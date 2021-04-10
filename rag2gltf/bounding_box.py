import copy
import math
from typing import List

import glm  # type: ignore

from node import Node
from parsing.rsm import Rsm
from utils import mat3tomat4, decode_string


class BoundingBox:
    def __init__(self):
        inf = float("inf")
        self.min = glm.vec3(inf, inf, inf)
        self.max = glm.vec3(-inf, -inf, -inf)
        self.offset = glm.vec3()
        self.range = glm.vec3()
        self.center = glm.vec3()


def calculate_model_bounding_box(rsm_version: int,
                                 nodes: List[Node]) -> BoundingBox:
    if rsm_version >= 0x200:
        # TODO
        return BoundingBox()

    _calculate_nodes_bounding_boxes(rsm_version, nodes)
    bbox = BoundingBox()
    for i in range(3):
        for node in nodes:
            if node.bbox is None:
                raise ValueError("Invalid bouding box")
            bbox.max[i] = max(bbox.max[i], node.bbox.max[i])
            bbox.min[i] = min(bbox.min[i], node.bbox.min[i])

        bbox.offset[i] = (bbox.max[i] + bbox.min[i]) / 2.0
        bbox.range[i] = (bbox.max[i] - bbox.min[i]) / 2.0
        bbox.center[i] = bbox.min[i] + bbox.range[i]

    return bbox


def _calculate_nodes_bounding_boxes(version: int, nodes: List[Node]) -> None:
    if len(nodes) == 0:
        return

    for node in nodes:
        if node.parent is None:
            _calculate_node_bounding_box(version, len(nodes), node)


def _calculate_node_bounding_box(
    version: int,
    node_count: int,
    node: Node,
    matrix: glm.mat4 = glm.mat4(1.0)) -> None:
    parent = node.parent
    rsm_node = node.impl
    bbox = BoundingBox()

    if parent is not None:
        matrix = glm.translate(matrix, glm.vec3(rsm_node.info.position))

    if rsm_node.rot_key_count == 0:
        if rsm_node.info.rotation_angle > 0.01:
            matrix = glm.rotate(
                matrix,
                glm.radians(rsm_node.info.rotation_angle * 180.0 / math.pi),
                glm.vec3(rsm_node.info.rotation_axis))
    else:
        quaternion = glm.quat(*rsm_node.rot_key_frames[0].quaternion)
        matrix *= glm.mat4_cast(glm.normalize(quaternion))

    matrix = glm.scale(matrix, glm.vec3(rsm_node.info.scale))

    local_matrix = copy.copy(matrix)

    offset_matrix = mat3tomat4(rsm_node.info.offset_matrix)
    if node_count > 1:
        local_matrix = glm.translate(local_matrix,
                                     glm.vec3(rsm_node.info.offset_vector))
    local_matrix *= offset_matrix

    for i in range(rsm_node.mesh_vertex_count):
        x = rsm_node.mesh_vertices[i].position[0]
        y = rsm_node.mesh_vertices[i].position[1]
        z = rsm_node.mesh_vertices[i].position[2]

        v = glm.vec3()
        v[0] = local_matrix[0][0] * x + local_matrix[1][0] * y + local_matrix[
            2][0] * z + local_matrix[3][0]
        v[1] = local_matrix[0][1] * x + local_matrix[1][1] * y + local_matrix[
            2][1] * z + local_matrix[3][1]
        v[2] = local_matrix[0][2] * x + local_matrix[1][2] * y + local_matrix[
            2][2] * z + local_matrix[3][2]

        for j in range(3):
            bbox.min[j] = min(v[j], bbox.min[j])
            bbox.max[j] = max(v[j], bbox.max[j])

    for i in range(3):
        bbox.offset[i] = (bbox.max[i] + bbox.min[i]) / 2.0
        bbox.range[i] = (bbox.max[i] - bbox.min[i]) / 2.0
        bbox.center[i] = bbox.min[i] + bbox.range[i]

    node.bbox = bbox
    for child in node.children:
        _calculate_node_bounding_box(version, node_count, child, matrix)
