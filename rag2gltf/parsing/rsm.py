# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
import kaitaistruct
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO
from enum import Enum

if parse_version(kaitaistruct.__version__) < parse_version('0.9'):
    raise Exception(
        "Incompatible Kaitai Struct Python API: 0.9 or later is required, but you have %s"
        % (kaitaistruct.__version__))


class Rsm(KaitaiStruct):
    class Shading(Enum):
        none = 0
        flat = 1
        smooth = 2

    def __init__(self, _io, _parent=None, _root=None):
        self._io = _io
        self._parent = _parent
        self._root = _root if _root else self
        self._read()

    def _read(self):
        self.magic = self._io.read_bytes(4)
        if not self.magic == b"\x47\x52\x53\x4D":
            raise kaitaistruct.ValidationNotEqualError(b"\x47\x52\x53\x4D",
                                                       self.magic, self._io,
                                                       u"/seq/0")
        self.version = self._io.read_u2be()
        self.animation_count = self._io.read_s4le()
        self.shading_type = KaitaiStream.resolve_enum(Rsm.Shading,
                                                      self._io.read_s4le())
        if self._root.version >= 260:
            self.alpha = self._io.read_u1()

        if self._root.version < 512:
            self.reserved = self._io.read_bytes(16)

        if self._root.version >= 514:
            self.frame_rate_per_second = self._io.read_f4le()

        if self._root.version <= 514:
            self.texture_count = self._io.read_s4le()
            _ = self.texture_count
            if not _ <= 1024:
                raise kaitaistruct.ValidationExprError(self.texture_count,
                                                       self._io, u"/seq/7")

        if self._root.version <= 514:
            self.texture_names = [None] * (self.texture_count)
            for i in range(self.texture_count):
                self.texture_names[i] = Rsm.String(self._io, self, self._root)

        if self._root.version >= 514:
            self.root_node_count = self._io.read_s4le()
            _ = self.root_node_count
            if not _ <= 1024:
                raise kaitaistruct.ValidationExprError(self.root_node_count,
                                                       self._io, u"/seq/9")

        if self._root.version >= 514:
            self.root_node_names = [None] * (self.root_node_count)
            for i in range(self.root_node_count):
                self.root_node_names[i] = Rsm.String(self._io, self,
                                                     self._root)

        if self._root.version < 512:
            self.root_node_name = Rsm.String(self._io, self, self._root)

        self.node_count = self._io.read_s4le()
        _ = self.node_count
        if not _ <= 1024:
            raise kaitaistruct.ValidationExprError(self.node_count, self._io,
                                                   u"/seq/12")
        self.nodes = [None] * (self.node_count)
        for i in range(self.node_count):
            self.nodes[i] = Rsm.Node(self._io, self, self._root)

        if self._root.version < 262:
            self.scale_key_count = self._io.read_s4le()
            _ = self.scale_key_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(self.scale_key_count,
                                                       self._io, u"/seq/14")

        if self._root.version < 262:
            self.scale_key_frames = [None] * (self.scale_key_count)
            for i in range(self.scale_key_count):
                self.scale_key_frames[i] = Rsm.ScaleKeyFrame(
                    self._io, self, self._root)

        if not (self._io.is_eof()):
            self.volume_box_count = self._io.read_s4le()
            _ = self.volume_box_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(self.volume_box_count,
                                                       self._io, u"/seq/16")

        if not (self._io.is_eof()):
            self.volume_boxes = [None] * (self.volume_box_count)
            for i in range(self.volume_box_count):
                self.volume_boxes[i] = Rsm.VolumeBox(self._io, self,
                                                     self._root)

    class TextureVertex(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            if self._root.version >= 258:
                self.color = self._io.read_u4le()

            self.position = [None] * (2)
            for i in range(2):
                self.position[i] = self._io.read_f4le()

    class FaceInfo(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            if self._root.version >= 514:
                self.length = self._io.read_s4le()

            self.mesh_vertex_ids = [None] * (3)
            for i in range(3):
                self.mesh_vertex_ids[i] = self._io.read_u2le()

            self.texture_vertex_ids = [None] * (3)
            for i in range(3):
                self.texture_vertex_ids[i] = self._io.read_u2le()

            self.texture_id = self._io.read_u2le()
            self.padding1 = self._io.read_u2le()
            self.two_sides = self._io.read_s4le()
            if self._root.version >= 258:
                self.smooth_group = self._io.read_s4le()

            if ((self._root.version >= 514) and (self.length > 24)):
                self.padding2 = self._io.read_bytes((self.length - 24))

    class PosKeyFrame(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.frame_id = self._io.read_s4le()
            self.position = [None] * (3)
            for i in range(3):
                self.position[i] = self._io.read_f4le()

            self.data = self._io.read_s4le()

    class RotKeyFrame(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.frame_id = self._io.read_s4le()
            self.quaternion = [None] * (4)
            for i in range(4):
                self.quaternion[i] = self._io.read_f4le()

    class TexAnimation(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.type = self._io.read_s4le()
            self.tex_frame_count = self._io.read_s4le()
            _ = self.tex_frame_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(
                    self.tex_frame_count, self._io,
                    u"/types/tex_animation/seq/1")
            self.tex_key_frames = [None] * (self.tex_frame_count)
            for i in range(self.tex_frame_count):
                self.tex_key_frames[i] = Rsm.TextureKeyFrame(
                    self._io, self, self._root)

    class TextureKeyFrame(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.frame_id = self._io.read_s4le()
            self.offset = self._io.read_f4le()

    class MeshVertex(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.position = [None] * (3)
            for i in range(3):
                self.position[i] = self._io.read_f4le()

    class NodeInfo(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.offset_matrix = [None] * (9)
            for i in range(9):
                self.offset_matrix[i] = self._io.read_f4le()

            self.offset_vector = [None] * (3)
            for i in range(3):
                self.offset_vector[i] = self._io.read_f4le()

            if self._root.version < 514:
                self.position = [None] * (3)
                for i in range(3):
                    self.position[i] = self._io.read_f4le()

            if self._root.version < 514:
                self.rotation_angle = self._io.read_f4le()

            if self._root.version < 514:
                self.rotation_axis = [None] * (3)
                for i in range(3):
                    self.rotation_axis[i] = self._io.read_f4le()

            if self._root.version < 514:
                self.scale = [None] * (3)
                for i in range(3):
                    self.scale[i] = self._io.read_f4le()

    class String(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            if self._root.version > 512:
                self.len = self._io.read_s4le()
                _ = self.len
                if not _ <= 1024:
                    raise kaitaistruct.ValidationExprError(
                        self.len, self._io, u"/types/string/seq/0")

            self.value = self._io.read_bytes(
                (self.len if self._root.version > 512 else 40))

    class VolumeBox(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.size = [None] * (3)
            for i in range(3):
                self.size[i] = self._io.read_f4le()

            self.position = [None] * (3)
            for i in range(3):
                self.position[i] = self._io.read_f4le()

            self.rotation = [None] * (3)
            for i in range(3):
                self.rotation[i] = self._io.read_f4le()

            if self._root.version >= 259:
                self.flag = self._io.read_s4le()

    class Node(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.name = Rsm.String(self._io, self, self._root)
            self.parent_name = Rsm.String(self._io, self, self._root)
            self.texture_count = self._io.read_s4le()
            _ = self.texture_count
            if not _ <= 1024:
                raise kaitaistruct.ValidationExprError(self.texture_count,
                                                       self._io,
                                                       u"/types/node/seq/2")
            if self._root.version >= 515:
                self.texture_names = [None] * (self.texture_count)
                for i in range(self.texture_count):
                    self.texture_names[i] = Rsm.String(self._io, self,
                                                       self._root)

            if self._root.version < 515:
                self.texture_ids = [None] * (self.texture_count)
                for i in range(self.texture_count):
                    self.texture_ids[i] = self._io.read_s4le()

            self.info = Rsm.NodeInfo(self._io, self, self._root)
            self.mesh_vertex_count = self._io.read_s4le()
            _ = self.mesh_vertex_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(self.mesh_vertex_count,
                                                       self._io,
                                                       u"/types/node/seq/6")
            self.mesh_vertices = [None] * (self.mesh_vertex_count)
            for i in range(self.mesh_vertex_count):
                self.mesh_vertices[i] = Rsm.MeshVertex(self._io, self,
                                                       self._root)

            self.texture_vertex_count = self._io.read_s4le()
            _ = self.texture_vertex_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(
                    self.texture_vertex_count, self._io, u"/types/node/seq/8")
            self.texture_vertices = [None] * (self.texture_vertex_count)
            for i in range(self.texture_vertex_count):
                self.texture_vertices[i] = Rsm.TextureVertex(
                    self._io, self, self._root)

            self.face_count = self._io.read_s4le()
            _ = self.face_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(self.face_count,
                                                       self._io,
                                                       u"/types/node/seq/10")
            self.faces = [None] * (self.face_count)
            for i in range(self.face_count):
                self.faces[i] = Rsm.FaceInfo(self._io, self, self._root)

            if self._root.version >= 262:
                self.scale_key_count = self._io.read_s4le()
                _ = self.scale_key_count
                if not _ <= 65536:
                    raise kaitaistruct.ValidationExprError(
                        self.scale_key_count, self._io, u"/types/node/seq/12")

            if self._root.version >= 262:
                self.scale_key_frames = [None] * (self.scale_key_count)
                for i in range(self.scale_key_count):
                    self.scale_key_frames[i] = Rsm.ScaleKeyFrame(
                        self._io, self, self._root)

            self.rot_key_count = self._io.read_s4le()
            _ = self.rot_key_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(self.rot_key_count,
                                                       self._io,
                                                       u"/types/node/seq/14")
            self.rot_key_frames = [None] * (self.rot_key_count)
            for i in range(self.rot_key_count):
                self.rot_key_frames[i] = Rsm.RotKeyFrame(
                    self._io, self, self._root)

            if self._root.version >= 514:
                self.pos_key_count = self._io.read_s4le()
                _ = self.pos_key_count
                if not _ <= 65536:
                    raise kaitaistruct.ValidationExprError(
                        self.pos_key_count, self._io, u"/types/node/seq/16")

            if self._root.version >= 514:
                self.pos_key_frames = [None] * (self.pos_key_count)
                for i in range(self.pos_key_count):
                    self.pos_key_frames[i] = Rsm.PosKeyFrame(
                        self._io, self, self._root)

            if self._root.version >= 515:
                self.animated_texture_count = self._io.read_s4le()
                _ = self.animated_texture_count
                if not _ <= 65536:
                    raise kaitaistruct.ValidationExprError(
                        self.animated_texture_count, self._io,
                        u"/types/node/seq/18")

            if self._root.version >= 515:
                self.animated_textures = [None] * (self.animated_texture_count)
                for i in range(self.animated_texture_count):
                    self.animated_textures[i] = Rsm.AnimatedTexture(
                        self._io, self, self._root)

    class ScaleKeyFrame(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.frame_id = self._io.read_s4le()
            self.scale = [None] * (3)
            for i in range(3):
                self.scale[i] = self._io.read_f4le()

            self.data = self._io.read_f4le()

    class AnimatedTexture(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.texture_id = self._io.read_s4le()
            self.animation_count = self._io.read_s4le()
            _ = self.animation_count
            if not _ <= 65536:
                raise kaitaistruct.ValidationExprError(
                    self.animation_count, self._io,
                    u"/types/animated_texture/seq/1")
            self.tex_animations = [None] * (self.animation_count)
            for i in range(self.animation_count):
                self.tex_animations[i] = Rsm.TexAnimation(
                    self._io, self, self._root)
