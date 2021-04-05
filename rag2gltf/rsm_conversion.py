import copy
import io
import math
import struct
import typing
from pathlib import Path, PureWindowsPath

import glm  # type: ignore
from gltflib import (  # type: ignore
    GLTF, GLTFModel, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer,
    BufferView, Accessor, AccessorType, BufferTarget, ComponentType,
    GLBResource, FileResource, Image, Sampler, Texture, Material,
    PBRMetallicRoughness, TextureInfo)
from kaitaistruct import KaitaiStream  # type: ignore

from bounding_box import calculate_model_bounding_box, BoundingBox
from image_conversion import convert_bmp_to_png, convert_tga_to_png
from node import Node as AbstractNode, extract_nodes
from parsing.rsm import Rsm
from utils import mat3tomat4, decode_string, rag_mat4_mul, decompose_matrix


def convert_rsm(rsm_file: str,
                data_folder: str = "data",
                glb: bool = False) -> None:
    rsm_file_path = Path(rsm_file)
    rsm_obj = _parse_rsm_file(rsm_file_path)

    (gltf_resources, gltf_images, gltf_textures, gltf_materials,
     tex_id_by_node) = _convert_textures(rsm_obj, Path(data_folder))

    nodes = extract_nodes(rsm_obj)
    (resources, gltf_buffers, gltf_buffer_views, gltf_accessors, gltf_meshes,
     gltf_nodes, gltf_root_nodes) = _convert_nodes(rsm_obj.version, nodes,
                                                   tex_id_by_node)
    gltf_resources += resources

    model = GLTFModel(
        asset=Asset(version='2.0', generator="rag2gltf"),
        scenes=[Scene(nodes=gltf_root_nodes)],
        nodes=gltf_nodes,
        meshes=gltf_meshes,
        buffers=gltf_buffers,
        bufferViews=gltf_buffer_views,
        accessors=gltf_accessors,
        images=gltf_images,
        samplers=[
            Sampler(
                magFilter=9729,  # LINEAR
                minFilter=9987,  # LINEAR_MIPMAP_LINEAR
                wrapS=33071,  # CLAMP_TO_EDGE
                wrapT=33071  # CLAMP_TO_EDGE
            )
        ],
        textures=gltf_textures,
        materials=gltf_materials)

    gltf = GLTF(model=model, resources=gltf_resources)
    if glb:
        gltf.export(rsm_file_path.with_suffix(".glb").name)
    else:
        gltf.export(rsm_file_path.with_suffix(".gltf").name)


def _parse_rsm_file(rsm_file_path: Path) -> Rsm:
    with rsm_file_path.open("rb") as f:
        stream = KaitaiStream(f)
        return Rsm(stream)


def _convert_textures(
    rsm_obj: Rsm, data_folder_path: Path
) -> typing.Tuple[typing.List[FileResource], typing.List[Image],
                  typing.List[Texture], typing.List[Material], typing.Dict[
                      str, list]]:
    gltf_resources = []
    gltf_images = []
    gltf_textures = []
    gltf_materials = []
    tex_id_by_node: typing.Dict[str, list] = {}

    if rsm_obj.version >= 0x203:
        texture_list: typing.List[Rsm.String] = []
        for node in rsm_obj.nodes:
            node_name = decode_string(node.name)
            tex_id_by_node[node_name] = []
            for texture in node.texture_names:
                try:
                    tex_id = texture_list.index(texture)
                    tex_id_by_node[node_name] += [tex_id]
                except ValueError:
                    texture_list.append(texture)
                    tex_id_by_node[node_name] += [len(texture_list) - 1]
    else:
        texture_list = rsm_obj.texture_names

    for tex_id, name_1252 in enumerate(texture_list):
        texture_path = PureWindowsPath(decode_string(name_1252))
        full_texture_path = data_folder_path / "texture" / texture_path
        texture_data: typing.Union[io.BytesIO, typing.BinaryIO]

        alpha_mode = "MASK"
        if texture_path.suffix.lower() == ".bmp":
            texture_data = convert_bmp_to_png(full_texture_path)
            dest_path = texture_path.with_suffix(".png")
        elif texture_path.suffix.lower() == ".tga":
            alpha_mode = "BLEND"
            texture_data = convert_tga_to_png(full_texture_path)
            dest_path = texture_path.with_suffix(".png")
        else:
            texture_data = full_texture_path.open("rb")
            dest_path = texture_path

        gltf_resources.append(
            FileResource(dest_path.name, data=texture_data.read()))
        gltf_images.append(Image(uri=dest_path.name, mimeType="image/png"))
        gltf_textures.append(Texture(sampler=0, source=tex_id))
        gltf_materials.append(
            Material(pbrMetallicRoughness=PBRMetallicRoughness(
                baseColorTexture=TextureInfo(index=tex_id),
                metallicFactor=0.0,
                roughnessFactor=1.0),
                     doubleSided=True,
                     alphaMode=alpha_mode))

    return (gltf_resources, gltf_images, gltf_textures, gltf_materials,
            tex_id_by_node)


def _convert_nodes(rsm_version: int, nodes: typing.List[AbstractNode],
                   tex_id_by_node: typing.Dict[str, list]):
    gltf_buffer_views = []
    gltf_accessors: typing.List[Accessor] = []
    gltf_meshes = []
    gltf_nodes: typing.List[Node] = []
    gltf_root_nodes = []

    model_bbox = calculate_model_bounding_box(rsm_version, nodes)
    for node in nodes:
        if node.parent is None:
            _compute_transform_matrices(rsm_version, node,
                                        len(nodes) == 0, model_bbox)

    nodes_children: typing.Dict[str, typing.List[int]] = {}
    vertex_bytearray = bytearray()
    tex_vertex_bytearray = bytearray()
    byteoffset = 0
    tex_byteoffset = 0
    for node_id, node in enumerate(nodes):
        rsm_node = node.impl
        node_name = decode_string(rsm_node.name)
        if node.parent is not None:
            parent_name = decode_string(node.parent.impl.name)
            if parent_name in nodes_children:
                nodes_children[parent_name] += [node_id]
            else:
                nodes_children[parent_name] = [node_id]

        if rsm_version >= 0x203:
            node_tex_ids = tex_id_by_node[node_name]
        else:
            node_tex_ids = rsm_node.texture_ids
        vertices_by_texture = _sort_vertices_by_texture(rsm_node, node_tex_ids)

        gltf_primitives = []
        for tex_id, vertices in vertices_by_texture.items():
            # Model vertices
            bytelen = _serialize_vertices(vertices[0], vertex_bytearray)
            # Texture vertices
            tex_bytelen = _serialize_vertices(vertices[1],
                                              tex_vertex_bytearray)

            (mins, maxs) = _calculate_vertices_bounds(vertices[0])
            (tex_mins, tex_maxs) = _calculate_vertices_bounds(vertices[1])

            gltf_buffer_views.append(
                BufferView(
                    buffer=0,  # Vertices
                    byteOffset=byteoffset,
                    byteLength=bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value))
            gltf_buffer_views.append(
                BufferView(
                    buffer=1,  # Texture vertices
                    byteOffset=tex_byteoffset,
                    byteLength=tex_bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value))

            buffer_view_id = len(gltf_accessors)
            gltf_accessors.append(
                Accessor(bufferView=buffer_view_id,
                         byteOffset=0,
                         componentType=ComponentType.FLOAT.value,
                         count=len(vertices[0]),
                         type=AccessorType.VEC3.value,
                         min=mins,
                         max=maxs))
            gltf_accessors.append(
                Accessor(bufferView=buffer_view_id + 1,
                         byteOffset=0,
                         componentType=ComponentType.FLOAT.value,
                         count=len(vertices[1]),
                         type=AccessorType.VEC2.value,
                         min=tex_mins,
                         max=tex_maxs))
            gltf_primitives.append(
                Primitive(attributes=Attributes(POSITION=buffer_view_id + 0,
                                                TEXCOORD_0=buffer_view_id + 1),
                          material=tex_id))

            byteoffset += bytelen
            tex_byteoffset += tex_bytelen

        gltf_meshes.append(Mesh(primitives=gltf_primitives))

        # Decompose matrix to TRS
        translation, rotation, scale = decompose_matrix(
            node.gltf_transform_matrix)

        gltf_nodes.append(
            Node(name=node_name,
                 mesh=node_id,
                 translation=translation.to_list() if translation else None,
                 rotation=rotation.to_list() if rotation else None,
                 scale=scale.to_list() if scale else None))
        if node.parent is None:
            gltf_root_nodes.append(node_id)

    # Register vertex buffers
    vtx_file_name = 'vertices.bin'
    tex_vtx_file_name = 'tex_vertices.bin'
    gltf_resources = [
        FileResource(vtx_file_name, data=vertex_bytearray),
        FileResource(tex_vtx_file_name, data=tex_vertex_bytearray)
    ]
    gltf_buffers = [
        Buffer(byteLength=byteoffset, uri=vtx_file_name),
        Buffer(byteLength=tex_byteoffset, uri=tex_vtx_file_name)
    ]

    # Update nodes' children
    for gltf_node in gltf_nodes:
        gltf_node.children = nodes_children.get(gltf_node.name)

    return (gltf_resources, gltf_buffers, gltf_buffer_views, gltf_accessors,
            gltf_meshes, gltf_nodes, gltf_root_nodes)


def _sort_vertices_by_texture(
        node: Rsm.Node,
        node_tex_ids: typing.List[int]) -> typing.Dict[int, list]:
    vertices_by_texture: typing.Dict[int, list] = {}

    for face_info in node.faces:
        v_ids = face_info.mesh_vertex_ids
        tv_ids = face_info.texture_vertex_ids
        vs = node.mesh_vertices
        t_vs = node.texture_vertices
        tex_id = node_tex_ids[face_info.texture_id]

        for i in range(3):
            vertex = vs[v_ids[i]]
            tex_vertex = t_vs[tv_ids[i]]
            if tex_id not in vertices_by_texture:
                vertices_by_texture[tex_id] = [[], []]

            vertices_by_texture[tex_id][0] += [tuple(vertex.position)]
            vertices_by_texture[tex_id][1] += [(tex_vertex.position[0],
                                                tex_vertex.position[1])]

    return vertices_by_texture


def _serialize_vertices(vertices: typing.List[typing.List[float]],
                        array: bytearray) -> int:
    initial_size = len(array)
    for vertex in vertices:
        for value in vertex:
            array.extend(struct.pack('f', value))

    return len(array) - initial_size


def _calculate_vertices_bounds(
    vertices: typing.List[typing.List[float]]
) -> typing.Tuple[typing.List[float], typing.List[float]]:
    vertex_len = len(vertices[0])
    min_v = [
        min([vertex[i] for vertex in vertices]) for i in range(vertex_len)
    ]
    max_v = [
        max([vertex[i] for vertex in vertices]) for i in range(vertex_len)
    ]
    return min_v, max_v


def _compute_transform_matrices(rsm_version: int, node: AbstractNode,
                                is_only_node: bool,
                                model_bbox: BoundingBox) -> None:
    if rsm_version >= 0x200:
        (node.local_transform_matrix,
         node.final_transform_matrix) = _generate_nodeview_matrix2(
             rsm_version, node)
        if node.parent:
            node.gltf_transform_matrix = glm.inverse(
                node.parent.final_transform_matrix
            ) * node.final_transform_matrix
        else:
            node.gltf_transform_matrix = node.final_transform_matrix
    elif rsm_version >= 0x100:
        node.gltf_transform_matrix = _generate_nodeview_matrix1(
            rsm_version, node, is_only_node, model_bbox)
    else:
        raise ValueError("Invalid RSM file version")

    for child in node.children:
        _compute_transform_matrices(rsm_version, child, is_only_node,
                                    model_bbox)


def _generate_nodeview_matrix1(rsm_version: int, node: AbstractNode,
                               is_only_node: bool,
                               model_bbox: BoundingBox) -> glm.mat4:
    rsm_node = node.impl
    # Model view
    # Translation
    nodeview_matrix = glm.mat4()
    if node.parent is None:
        # Z axis is in the opposite direction in glTF
        nodeview_matrix = glm.rotate(nodeview_matrix, math.pi,
                                     glm.vec3(1.0, 0.0, 0.0))
        if is_only_node:
            nodeview_matrix = glm.translate(
                nodeview_matrix,
                glm.vec3(-model_bbox.center[0] + model_bbox.range[0],
                         -model_bbox.max[1] + model_bbox.range[1],
                         -model_bbox.min[2]))
        else:
            nodeview_matrix = glm.translate(
                nodeview_matrix,
                glm.vec3(-model_bbox.center[0], -model_bbox.max[1],
                         -model_bbox.center[2]))
    else:
        nodeview_matrix = glm.rotate(nodeview_matrix, -math.pi / 2,
                                     glm.vec3(1.0, 0.0, 0.0))
        nodeview_matrix = glm.translate(nodeview_matrix,
                                        glm.vec3(rsm_node.info.position))

    # Figure out the initial rotation
    if len(rsm_node.rot_key_frames) == 0:
        # Static model
        if rsm_node.info.rotation_angle > 0.01:
            nodeview_matrix = glm.rotate(
                nodeview_matrix,
                glm.radians(rsm_node.info.rotation_angle * 180.0 / math.pi),
                glm.vec3(rsm_node.info.rotation_axis))
    else:
        # Animated model
        key_frame = rsm_node.rot_key_frames[0]
        quaternion = glm.normalize(
            glm.quat(key_frame.quaternion[3], key_frame.quaternion[0],
                     key_frame.quaternion[1], key_frame.quaternion[2]))
        nodeview_matrix *= glm.mat4_cast(quaternion)

    # Scaling
    nodeview_matrix = glm.scale(nodeview_matrix, glm.vec3(rsm_node.info.scale))

    # Node view
    if is_only_node:
        nodeview_matrix = glm.translate(nodeview_matrix, -model_bbox.range)
    elif node.parent is not None:
        nodeview_matrix = glm.translate(nodeview_matrix,
                                        glm.vec3(rsm_node.info.offset_vector))
    nodeview_matrix *= mat3tomat4(rsm_node.info.offset_matrix)

    return nodeview_matrix


def _generate_nodeview_matrix2(
        rsm_version: int,
        node: AbstractNode) -> typing.Tuple[glm.mat4, glm.mat4]:
    # Transformations which are inherited by children
    local_transform_matrix = glm.mat4()
    rsm_node = node.impl

    # Scaling
    if len(rsm_node.scale_key_frames) > 0:
        local_transform_matrix = glm.scale(
            local_transform_matrix,
            glm.vec3(rsm_node.scale_key_frames[0].scale))

    # Rotation
    if len(rsm_node.rot_key_frames) > 0:
        # Animated model
        key_frame = rsm_node.rot_key_frames[0]
        quaternion = glm.quat(key_frame.quaternion[3], key_frame.quaternion[0],
                              key_frame.quaternion[1], key_frame.quaternion[2])
        local_transform_matrix *= glm.mat4_cast(quaternion)
    else:
        # Static model
        local_transform_matrix = rag_mat4_mul(
            local_transform_matrix, mat3tomat4(rsm_node.info.offset_matrix))
        if node.parent:
            parent_offset_matrix = mat3tomat4(
                node.parent.impl.info.offset_matrix)
            local_transform_matrix = rag_mat4_mul(
                local_transform_matrix, glm.inverse(parent_offset_matrix))

    # Translation
    if rsm_version >= 0x203 and len(rsm_node.pos_key_frames) > 0:
        key_frame = rsm_node.pos_key_frames[0]
        position = glm.vec3(key_frame.position)
    elif node.parent:
        position = glm.vec3(rsm_node.info.offset_vector) - \
            glm.vec3(node.parent.impl.info.offset_vector)
        parent_offset_matrix = mat3tomat4(node.parent.impl.info.offset_matrix)
        position = glm.vec3(
            glm.inverse(parent_offset_matrix) * glm.vec4(position, 1.0))
    else:
        position = glm.vec3(rsm_node.info.offset_vector)

    # Transformations which are applied only to this node
    final_transform_matrix = copy.copy(local_transform_matrix)
    # Reset translation transformation to `position`
    final_transform_matrix[3] = glm.vec4(position, 1.0)

    # Inherit transformations from ancestors
    parent = node.parent
    while parent:
        final_transform_matrix = rag_mat4_mul(final_transform_matrix,
                                              parent.local_transform_matrix)
        parent = parent.parent

    if node.parent:
        parent_translation = glm.vec3(node.parent.final_transform_matrix[3])
        final_transform_matrix[3] += glm.vec4(parent_translation, 0.0)

    return (local_transform_matrix, final_transform_matrix)