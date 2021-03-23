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
from parsing.rsm import Rsm
from utils import mat3tomat4, decode_zstr


def convert_rsm(rsm_file: str,
                data_folder: str = "data",
                glb: bool = False) -> None:
    rsm_file_path = Path(rsm_file)
    rsm_obj = _parse_rsm_file(rsm_file_path)

    (gltf_resources, gltf_images, gltf_textures,
     gltf_materials) = _convert_textures(rsm_obj, Path(data_folder))

    (resources, gltf_buffers, gltf_buffer_views, gltf_accessors, gltf_meshes,
     gltf_nodes, gltf_root_nodes) = _convert_nodes(rsm_obj)
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
                  typing.List[Texture], typing.List[Material]]:
    gltf_resources = []
    gltf_images = []
    gltf_textures = []
    gltf_materials = []

    for tex_id, name_1252 in enumerate(rsm_obj.texture_names):
        texture_path = PureWindowsPath(decode_zstr(name_1252))
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

    return (gltf_resources, gltf_images, gltf_textures, gltf_materials)


def _convert_nodes(rsm_obj: Rsm):
    gltf_buffer_views = []
    gltf_accessors: typing.List[Accessor] = []
    gltf_meshes = []
    gltf_nodes = []
    gltf_root_nodes = []

    model_bbox = calculate_model_bounding_box(rsm_obj)
    nodes_children: typing.Dict[str, typing.List[int]] = {}
    vertex_bytearray = bytearray()
    tex_vertex_bytearray = bytearray()
    byteoffset = 0
    tex_byteoffset = 0
    for node_id, node in enumerate(rsm_obj.nodes):
        node_name = decode_zstr(node.name)
        parent_node_name = decode_zstr(node.parent_name)
        is_root_node = len(parent_node_name) == 0
        if not is_root_node:
            if parent_node_name in nodes_children:
                nodes_children[parent_node_name] += [node_id]
            else:
                nodes_children[parent_node_name] = [node_id]

        vertices_by_texture = _sort_vertices_by_texture(node)
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

        nodeview_matrix = _generate_nodeview_matrix(node, is_root_node,
                                                    rsm_obj.node_count == 1,
                                                    model_bbox)
        gltf_nodes.append(
            Node(name=node_name,
                 mesh=node_id,
                 matrix=sum(nodeview_matrix.to_list(), [])))
        if is_root_node:
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
    for node in gltf_nodes:
        node.children = nodes_children.get(node.name)

    return (gltf_resources, gltf_buffers, gltf_buffer_views, gltf_accessors,
            gltf_meshes, gltf_nodes, gltf_root_nodes)


def _sort_vertices_by_texture(node: Rsm.Node) -> typing.Dict[int, list]:
    vertices_by_texture: typing.Dict[int, list] = {
        tex_id: [[], []]
        for tex_id in node.texture_ids
    }

    for face_info in node.faces_info:
        v_ids = face_info.node_vertex_ids
        tv_ids = face_info.texture_vertex_ids
        vs = node.node_vertices
        t_vs = node.texture_vertices
        tex_id = node.texture_ids[face_info.texture_id]
        for i in range(3):
            vertex = vs[v_ids[i]]
            tex_vertex = t_vs[tv_ids[i]]

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


def _generate_nodeview_matrix(node: Rsm.Node, is_root_node: bool,
                              is_only_node: bool,
                              model_bbox: BoundingBox) -> glm.mat4:
    # Model view
    # Translation
    nodeview_matrix = glm.mat4(1.0)
    if is_root_node:
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
                                        glm.vec3(node.info.position))

    # Figure out the initial rotation
    if node.rot_key_count == 0:
        # Static model
        if node.info.rotation_angle > 0.01:
            nodeview_matrix = glm.rotate(
                nodeview_matrix,
                glm.radians(node.info.rotation_angle * 180.0 / math.pi),
                glm.vec3(node.info.rotation_axis))
    else:
        # Animated model
        quaternion = glm.quat(node.rot_key_frames[0].quaternion)
        nodeview_matrix *= glm.mat4(quaternion)

    # Scaling
    nodeview_matrix = glm.scale(nodeview_matrix, glm.vec3(node.info.scale))

    # Node view
    if is_only_node:
        nodeview_matrix = glm.translate(nodeview_matrix, -model_bbox.range)
    elif not is_root_node:
        nodeview_matrix = glm.translate(nodeview_matrix,
                                        glm.vec3(node.info.offset_vector))
    nodeview_matrix *= mat3tomat4(node.info.offset_matrix)

    return nodeview_matrix
