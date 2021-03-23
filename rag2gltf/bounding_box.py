import copy
import math
import typing

import glm  # type: ignore

from parsing.rsm import Rsm
from utils import mat3tomat4, decode_zstr


class BoundingBox:
    def __init__(self):
        inf = float("inf")
        self.min = glm.vec3(inf, inf, inf)
        self.max = glm.vec3(-inf, -inf, -inf)
        self.offset = glm.vec3()
        self.range = glm.vec3()
        self.center = glm.vec3()


def calculate_model_bounding_box(rsm_obj: Rsm) -> BoundingBox:
    nodes_bboxes = _calculate_nodes_bounding_boxes(rsm_obj.nodes)
    bbox = BoundingBox()
    for i in range(3):
        for j in range(rsm_obj.node_count):
            bbox.max[i] = max(bbox.max[i], nodes_bboxes[j].max[i])
            bbox.min[i] = min(bbox.min[i], nodes_bboxes[j].min[i])

        bbox.offset[i] = (bbox.max[i] + bbox.min[i]) / 2.0
        bbox.range[i] = (bbox.max[i] - bbox.min[i]) / 2.0
        bbox.center[i] = bbox.min[i] + bbox.range[i]

    return bbox


def _calculate_nodes_bounding_boxes(
        nodes: typing.List[Rsm.Node]) -> typing.List[BoundingBox]:
    return _calculate_node_bounding_box(nodes, 0, [BoundingBox()] * len(nodes))


def _calculate_node_bounding_box(
    nodes: typing.List[Rsm.Node],
    node_id: int,
    bboxes: typing.List[BoundingBox],
    matrix: glm.mat4 = glm.mat4(1.0)
) -> typing.List[BoundingBox]:
    parent_name = decode_zstr(nodes[node_id].parent_name)
    node = nodes[node_id]
    bbox = BoundingBox()

    if len(parent_name) == 0:
        matrix = glm.mat4(1.0)
    else:
        matrix = glm.translate(matrix, glm.vec3(node.info.position))

    if node.rot_key_count == 0:
        if node.info.rotation_angle > 0.01:
            matrix = glm.rotate(
                matrix,
                glm.radians(node.info.rotation_angle * 180.0 / math.pi),
                glm.vec3(node.info.rotation_axis))
    else:
        quaternion = glm.quat(node.rot_key_frames[0].quaternion)
        matrix *= glm.mat4(glm.normalize(quaternion))

    matrix = glm.scale(matrix, glm.vec3(node.info.scale))

    local_matrix = copy.copy(matrix)

    offset_matrix = mat3tomat4(node.info.offset_matrix)
    if len(nodes) > 1:
        local_matrix = glm.translate(local_matrix,
                                     glm.vec3(node.info.offset_vector))
    local_matrix *= offset_matrix

    for i in range(node.node_vertex_count):
        x = node.node_vertices[i].position[0]
        y = node.node_vertices[i].position[1]
        z = node.node_vertices[i].position[2]

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

    bboxes[node_id] = bbox
    node_name = decode_zstr(node.name)
    for other_node_id in range(len(nodes)):
        parent_name = decode_zstr(nodes[other_node_id].parent_name)
        if parent_name == node_name:
            bboxes = _calculate_node_bounding_box(nodes, other_node_id, bboxes,
                                                  matrix)

    return bboxes