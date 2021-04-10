import copy
import io
import logging
import math
import multiprocessing as mp
import struct
import sys
from functools import partial
from pathlib import Path, PureWindowsPath
from typing import List, Dict, Union, Tuple, Optional, BinaryIO

import glm  # type: ignore
from gltflib import (  # type: ignore
    GLTF, GLTFModel, Asset, Scene, Node, Mesh, Primitive, Attributes, Buffer,
    BufferView, Accessor, AccessorType, BufferTarget, ComponentType,
    FileResource, Image, Sampler, Texture, Material, PBRMetallicRoughness,
    TextureInfo, Animation, AnimationSampler, Channel, Target)
from kaitaistruct import KaitaiStream, ValidationNotEqualError  # type: ignore

from .bounding_box import calculate_model_bounding_box, BoundingBox
from .image_conversion import convert_bmp_to_png, convert_tga_to_png
from .node import Node as AbstractNode, extract_nodes
from .parsing.rsm import Rsm
from .utils import (mat3tomat4, decode_string, rag_mat4_mul, decompose_matrix,
                    serialize_floats)

_LOGGER = logging.getLogger("rag2gltf")


def convert_rsm(rsm_file: str,
                data_folder: str = "data",
                glb: bool = False) -> None:
    logging.basicConfig(level=logging.INFO)

    _LOGGER.info(f"Converting RSM file '{rsm_file}'")
    rsm_file_path = Path(rsm_file)
    try:
        rsm_obj = _parse_rsm_file(rsm_file_path)
    except FileNotFoundError:
        _LOGGER.error(f"'{rsm_file_path}' isn't a file or doesn't exist")
        sys.exit(1)
    except ValidationNotEqualError as ex:
        _LOGGER.error(f"Invalid RSM file: {ex}")
        sys.exit(1)

    gltf_model = GLTFModel(
        asset=Asset(version='2.0', generator="rag2gltf"),
        samplers=[
            Sampler(
                magFilter=9729,  # LINEAR
                minFilter=9987,  # LINEAR_MIPMAP_LINEAR
                wrapS=33071,  # CLAMP_TO_EDGE
                wrapT=33071  # CLAMP_TO_EDGE
            )
        ],
        nodes=[],
        meshes=[],
        buffers=[],
        bufferViews=[],
        accessors=[],
        images=[],
        textures=[],
        materials=[])

    gltf_resources: List[FileResource] = []
    _LOGGER.info("Converting textures ...")
    try:
        resources, tex_id_by_node = _convert_textures(rsm_obj,
                                                      Path(data_folder),
                                                      gltf_model)
    except FileNotFoundError as ex:
        _LOGGER.error(f"Cannot find texture file: {ex}")
        sys.exit(1)
    gltf_resources += resources

    _LOGGER.info("Converting 3D model ...")
    nodes = extract_nodes(rsm_obj)
    resources, root_nodes = _convert_nodes(rsm_obj.version, nodes,
                                           tex_id_by_node, gltf_model)
    gltf_model.scenes = [Scene(nodes=root_nodes)]
    gltf_resources += resources

    # Convert animations
    if rsm_obj.version >= 0x202:
        fps = rsm_obj.frame_rate_per_second
    else:
        fps = None

    _LOGGER.info("Converting animations ...")
    resources = _convert_animations(rsm_obj.version, fps, nodes, gltf_model)
    gltf_resources += resources

    if glb:
        destination_path = rsm_file_path.with_suffix(".glb").name
    else:
        destination_path = rsm_file_path.with_suffix(".gltf").name

    gltf = GLTF(model=gltf_model, resources=gltf_resources)
    gltf.export(destination_path)
    _LOGGER.info(f"Converted model has been saved as '{destination_path}'")

    sys.exit()


def _parse_rsm_file(rsm_file_path: Path) -> Rsm:
    with rsm_file_path.open("rb") as f:
        stream = KaitaiStream(f)
        return Rsm(stream)


def _convert_textures(
        rsm_obj: Rsm, data_folder_path: Path,
        gltf_model: GLTFModel) -> Tuple[List[FileResource], Dict[str, list]]:
    tex_id_by_node: Dict[str, list] = {}

    if rsm_obj.version >= 0x203:
        texture_list: List[Rsm.String] = []
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

    # Convert textures in parallel
    with mp.Pool() as pool:
        convert_texture = partial(_convert_texture,
                                  data_folder_path=data_folder_path)
        result_list = pool.map(
            convert_texture,
            map(lambda t: (t[0], decode_string(t[1])),
                enumerate(texture_list)))

    gltf_resources = []
    for material, texture, image, resource in result_list:
        gltf_resources.append(resource)
        gltf_model.images.append(image)
        gltf_model.textures.append(texture)
        gltf_model.materials.append(material)

    return gltf_resources, tex_id_by_node


def _convert_texture(
        tex_info: Tuple[int, str], data_folder_path: Path
) -> Tuple[Material, Texture, Image, FileResource]:
    tex_id, texture_name = tex_info
    texture_path = PureWindowsPath(texture_name)
    full_texture_path = data_folder_path / "texture" / texture_path

    texture_data: Union[io.BytesIO, BinaryIO]
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

    return (Material(pbrMetallicRoughness=PBRMetallicRoughness(
        baseColorTexture=TextureInfo(index=tex_id),
        metallicFactor=0.0,
        roughnessFactor=1.0),
                     doubleSided=True,
                     alphaMode=alpha_mode), Texture(sampler=0, source=tex_id),
            Image(uri=dest_path.name, mimeType="image/png"),
            FileResource(dest_path.name, data=texture_data.read()))


def _convert_nodes(
        rsm_version: int, nodes: List[AbstractNode],
        tex_id_by_node: Dict[str, list],
        gltf_model: GLTFModel) -> Tuple[List[FileResource], List[int]]:
    root_nodes = []

    model_bbox = calculate_model_bounding_box(rsm_version, nodes)
    for node in nodes:
        if node.parent is None:
            _compute_transform_matrices(rsm_version, node,
                                        len(nodes) == 0, model_bbox)

    nodes_children: Dict[str, List[int]] = {}
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

            gltf_model.bufferViews.append(
                BufferView(
                    buffer=0,  # Vertices
                    byteOffset=byteoffset,
                    byteLength=bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value))
            gltf_model.bufferViews.append(
                BufferView(
                    buffer=1,  # Texture vertices
                    byteOffset=tex_byteoffset,
                    byteLength=tex_bytelen,
                    target=BufferTarget.ARRAY_BUFFER.value))

            buffer_view_id = len(gltf_model.accessors)
            gltf_model.accessors.append(
                Accessor(bufferView=buffer_view_id,
                         byteOffset=0,
                         componentType=ComponentType.FLOAT.value,
                         count=len(vertices[0]),
                         type=AccessorType.VEC3.value,
                         min=mins,
                         max=maxs))
            gltf_model.accessors.append(
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

        gltf_model.meshes.append(Mesh(primitives=gltf_primitives))

        # Decompose matrix to TRS
        translation, rotation, scale = decompose_matrix(
            node.gltf_transform_matrix)

        gltf_model.nodes.append(
            Node(name=node_name,
                 mesh=node_id,
                 translation=translation.to_list() if translation else None,
                 rotation=rotation.to_list() if rotation else None,
                 scale=scale.to_list() if scale else None))
        if node.parent is None:
            root_nodes.append(node_id)

    # Register vertex buffers
    vtx_file_name = 'vertices.bin'
    tex_vtx_file_name = 'tex_vertices.bin'
    gltf_resources = [
        FileResource(vtx_file_name, data=vertex_bytearray),
        FileResource(tex_vtx_file_name, data=tex_vertex_bytearray)
    ]
    gltf_model.buffers = [
        Buffer(byteLength=byteoffset, uri=vtx_file_name),
        Buffer(byteLength=tex_byteoffset, uri=tex_vtx_file_name)
    ]

    # Update nodes' children
    for gltf_node in gltf_model.nodes:
        gltf_node.children = nodes_children.get(gltf_node.name)

    return gltf_resources, root_nodes


def _sort_vertices_by_texture(node: Rsm.Node,
                              node_tex_ids: List[int]) -> Dict[int, list]:
    vertices_by_texture: Dict[int, list] = {}

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


def _serialize_vertices(vertices: List[List[float]], array: bytearray) -> int:
    initial_size = len(array)
    for vertex in vertices:
        for value in vertex:
            array.extend(struct.pack('f', value))

    return len(array) - initial_size


def _calculate_vertices_bounds(
        vertices: List[List[float]]) -> Tuple[List[float], List[float]]:
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
        quaternion = glm.quat(key_frame.quaternion[3], key_frame.quaternion[0],
                              key_frame.quaternion[1], key_frame.quaternion[2])
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
        rsm_version: int, node: AbstractNode) -> Tuple[glm.mat4, glm.mat4]:
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


def _convert_animations(rsm_version: int,
                        frame_rate_per_second: Optional[float],
                        nodes: List[AbstractNode],
                        gltf_model: GLTFModel) -> List[FileResource]:
    gltf_resources = []

    if frame_rate_per_second:
        delay_between_frames = 1.0 / frame_rate_per_second
    else:
        delay_between_frames = 1.0 / 1000.0

    model_anim = Animation(name="animation", samplers=[], channels=[])
    input_buffer_id = None
    input_stream = io.BytesIO()
    rot_buffer_id = None
    rot_output_stream = io.BytesIO()
    scale_buffer_id = None
    scale_output_stream = io.BytesIO()
    pos_buffer_id = None
    pos_output_stream = io.BytesIO()

    for node_id, node in enumerate(nodes):
        rsm_node = node.impl
        node_name = decode_string(rsm_node.name)

        # Rotation
        rotation_frame_count = len(rsm_node.rot_key_frames)
        if rotation_frame_count > 0:
            if input_buffer_id is None:
                input_buffer_id = len(gltf_model.buffers)
                gltf_model.buffers.append(Buffer(byteLength=0))
            if rot_buffer_id is None:
                rot_buffer_id = len(gltf_model.buffers)
                gltf_model.buffers.append(Buffer(byteLength=0))

            input_values = [
                delay_between_frames * rot_frame.frame_id
                for rot_frame in rsm_node.rot_key_frames
            ]

            input_view_offset = input_stream.tell()
            input_written = serialize_floats(input_values, input_stream)
            output_view_offset = rot_output_stream.tell()
            output_written = 0
            for frame in rsm_node.rot_key_frames:
                if rsm_version < 0x200:
                    gltf_quat = [
                        frame.quaternion[0],
                        frame.quaternion[2],
                        frame.quaternion[1],
                        frame.quaternion[3],
                    ]
                else:
                    gltf_quat = frame.quaternion
                for value in gltf_quat:
                    output_written += rot_output_stream.write(
                        struct.pack('f', value))

            curr_buffer_view_id = len(gltf_model.bufferViews)
            gltf_model.bufferViews += [
                BufferView(buffer=input_buffer_id,
                           byteOffset=input_view_offset,
                           byteLength=input_written),
                BufferView(buffer=rot_buffer_id,
                           byteOffset=output_view_offset,
                           byteLength=output_written)
            ]
            curr_accessor_id = len(gltf_model.accessors)
            gltf_model.accessors += [
                Accessor(bufferView=curr_buffer_view_id,
                         byteOffset=0,
                         componentType=ComponentType.FLOAT.value,
                         count=rotation_frame_count,
                         type=AccessorType.SCALAR.value,
                         min=[min(input_values)],
                         max=[max(input_values)]),
                Accessor(bufferView=curr_buffer_view_id + 1,
                         byteOffset=0,
                         componentType=ComponentType.FLOAT.value,
                         count=rotation_frame_count,
                         type=AccessorType.VEC4.value)
            ]

            rot_sampler = AnimationSampler(input=curr_accessor_id,
                                           output=curr_accessor_id + 1)
            sampler_id = len(model_anim.samplers)
            rot_channel = Channel(sampler=sampler_id,
                                  target=Target(path="rotation", node=node_id))

            model_anim.samplers.append(rot_sampler)
            model_anim.channels.append(rot_channel)

        # Scale
        if rsm_version >= 0x106:
            scale_frame_count = len(rsm_node.scale_key_frames)
            if scale_frame_count > 0:
                if input_buffer_id is None:
                    input_buffer_id = len(gltf_model.buffers)
                    gltf_model.buffers.append(Buffer(byteLength=0))
                if scale_buffer_id is None:
                    scale_buffer_id = len(gltf_model.buffers)
                    gltf_model.buffers.append(Buffer(byteLength=0))

                input_values = [
                    delay_between_frames * scale_frame.frame_id
                    for scale_frame in rsm_node.scale_key_frames
                ]

                input_view_offset = input_stream.tell()
                input_written = serialize_floats(input_values, input_stream)
                output_view_offset = scale_output_stream.tell()
                output_written = 0
                for frame in rsm_node.scale_key_frames:
                    for value in frame.scale:
                        output_written += scale_output_stream.write(
                            struct.pack('f', value))

                curr_buffer_view_id = len(gltf_model.bufferViews)
                gltf_model.bufferViews += [
                    BufferView(buffer=input_buffer_id,
                               byteOffset=input_view_offset,
                               byteLength=input_written),
                    BufferView(buffer=scale_buffer_id,
                               byteOffset=output_view_offset,
                               byteLength=output_written)
                ]
                curr_accessor_id = len(gltf_model.accessors)
                gltf_model.accessors += [
                    Accessor(bufferView=curr_buffer_view_id,
                             byteOffset=0,
                             componentType=ComponentType.FLOAT.value,
                             count=scale_frame_count,
                             type=AccessorType.SCALAR.value,
                             min=[min(input_values)],
                             max=[max(input_values)]),
                    Accessor(bufferView=curr_buffer_view_id + 1,
                             byteOffset=0,
                             componentType=ComponentType.FLOAT.value,
                             count=scale_frame_count,
                             type=AccessorType.VEC3.value)
                ]

                scale_sampler = AnimationSampler(input=curr_accessor_id,
                                                 output=curr_accessor_id + 1)
                sampler_id = len(model_anim.samplers)
                scale_channel = Channel(sampler=sampler_id,
                                        target=Target(path="scale",
                                                      node=node_id))

                model_anim.samplers.append(scale_sampler)
                model_anim.channels.append(scale_channel)

        # Translation
        if rsm_version >= 0x203:
            translation_frame_count = len(rsm_node.pos_key_frames)
            if translation_frame_count > 0:
                if input_buffer_id is None:
                    input_buffer_id = len(gltf_model.buffers)
                    gltf_model.buffers.append(Buffer(byteLength=0))
                if pos_buffer_id is None:
                    pos_buffer_id = len(gltf_model.buffers)
                    gltf_model.buffers.append(Buffer(byteLength=0))

                input_values = [
                    delay_between_frames * pos_frame.frame_id
                    for pos_frame in rsm_node.pos_key_frames
                ]

                input_view_offset = input_stream.tell()
                input_written = serialize_floats(input_values, input_stream)
                output_view_offset = pos_output_stream.tell()
                output_written = 0
                for frame in rsm_node.pos_key_frames:
                    for value in frame.position:
                        output_written += pos_output_stream.write(
                            struct.pack('f', value))

                curr_buffer_view_id = len(gltf_model.bufferViews)
                gltf_model.bufferViews += [
                    BufferView(buffer=input_buffer_id,
                               byteOffset=input_view_offset,
                               byteLength=input_written),
                    BufferView(buffer=pos_buffer_id,
                               byteOffset=output_view_offset,
                               byteLength=output_written)
                ]
                curr_accessor_id = len(gltf_model.accessors)
                gltf_model.accessors += [
                    Accessor(bufferView=curr_buffer_view_id,
                             byteOffset=0,
                             componentType=ComponentType.FLOAT.value,
                             count=translation_frame_count,
                             type=AccessorType.SCALAR.value,
                             min=[min(input_values)],
                             max=[max(input_values)]),
                    Accessor(bufferView=curr_buffer_view_id + 1,
                             byteOffset=0,
                             componentType=ComponentType.FLOAT.value,
                             count=translation_frame_count,
                             type=AccessorType.VEC3.value)
                ]

                pos_sampler = AnimationSampler(input=curr_accessor_id,
                                               output=curr_accessor_id + 1)
                sampler_id = len(model_anim.samplers)
                pos_channel = Channel(sampler=sampler_id,
                                      target=Target(path="translation",
                                                    node=node_id))

                model_anim.samplers.append(pos_sampler)
                model_anim.channels.append(pos_channel)

    if input_buffer_id:
        # Add input data
        input_stream.seek(0)
        input_data = input_stream.read()
        input_file_name = "anim_in.bin"
        gltf_resources.append(FileResource(input_file_name, data=input_data))
        gltf_model.buffers[input_buffer_id].uri = input_file_name
        gltf_model.buffers[input_buffer_id].byteLength = len(input_data)

        # Add rotation data
        if rot_buffer_id:
            rot_output_stream.seek(0)
            rot_data = rot_output_stream.read()
            rot_file_name = 'anim_rot.bin'
            gltf_resources.append(FileResource(rot_file_name, data=rot_data))
            gltf_model.buffers[rot_buffer_id].uri = rot_file_name
            gltf_model.buffers[rot_buffer_id].byteLength = len(rot_data)

        # Add scale data
        if scale_buffer_id:
            scale_output_stream.seek(0)
            scale_data = scale_output_stream.read()
            scale_file_name = 'anim_scale.bin'
            gltf_resources.append(
                FileResource(scale_file_name, data=scale_data))
            gltf_model.buffers[scale_buffer_id].uri = scale_file_name
            gltf_model.buffers[scale_buffer_id].byteLength = len(scale_data)

        # Add tanslation data
        if pos_buffer_id:
            pos_output_stream.seek(0)
            pos_data = pos_output_stream.read()
            pos_file_name = 'anim_pos.bin'
            gltf_resources.append(FileResource(pos_file_name, data=pos_data))
            gltf_model.buffers[pos_buffer_id].uri = pos_file_name
            gltf_model.buffers[pos_buffer_id].byteLength = len(pos_data)

        gltf_model.animations = [model_anim]

    return gltf_resources