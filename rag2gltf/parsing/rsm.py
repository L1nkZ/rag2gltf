# This is a generated file! Please edit source .ksy file and use kaitai-struct-compiler to rebuild

from pkg_resources import parse_version
import kaitaistruct  # type: ignore
from kaitaistruct import KaitaiStruct, KaitaiStream, BytesIO  # type: ignore
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
        self.alpha = self._io.read_u1()
        self.reserved = self._io.read_bytes(16)
        self.texture_count = self._io.read_s4le()
        self.texture_names = [None] * (self.texture_count)
        for i in range(self.texture_count):
            self.texture_names[i] = self._io.read_bytes(40)

        self.main_node_name = self._io.read_bytes(40)
        self.node_count = self._io.read_s4le()
        self.nodes = [None] * (self.node_count)
        for i in range(self.node_count):
            self.nodes[i] = Rsm.Node(self._io, self, self._root)

        if self._root.version < 261:
            self.pos_key_count = self._io.read_s4le()

        if self._root.version < 261:
            self.pos_key_frames = [None] * (self.pos_key_count)
            for i in range(self.pos_key_count):
                self.pos_key_frames[i] = Rsm.PosKeyFrame(
                    self._io, self, self._root)

        self.volume_box_count = self._io.read_s4le()
        self.volume_boxes = [None] * (self.volume_box_count)
        for i in range(self.volume_box_count):
            self.volume_boxes[i] = Rsm.VolumeBox(self._io, self, self._root)

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
            self.node_vertex_ids = [None] * (3)
            for i in range(3):
                self.node_vertex_ids[i] = self._io.read_u2le()

            self.texture_vertex_ids = [None] * (3)
            for i in range(3):
                self.texture_vertex_ids[i] = self._io.read_u2le()

            self.texture_id = self._io.read_u2le()
            self.padding = self._io.read_u2le()
            self.two_sides = self._io.read_s4le()
            if self._root.version >= 258:
                self.smooth_group = self._io.read_s4le()

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

            self.position = [None] * (3)
            for i in range(3):
                self.position[i] = self._io.read_f4le()

            self.rotation_angle = self._io.read_f4le()
            self.rotation_axis = [None] * (3)
            for i in range(3):
                self.rotation_axis[i] = self._io.read_f4le()

            self.scale = [None] * (3)
            for i in range(3):
                self.scale[i] = self._io.read_f4le()

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

    class NodeVertex(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.position = [None] * (3)
            for i in range(3):
                self.position[i] = self._io.read_f4le()

    class Node(KaitaiStruct):
        def __init__(self, _io, _parent=None, _root=None):
            self._io = _io
            self._parent = _parent
            self._root = _root if _root else self
            self._read()

        def _read(self):
            self.name = self._io.read_bytes(40)
            self.parent_name = self._io.read_bytes(40)
            self.texture_count = self._io.read_s4le()
            self.texture_ids = [None] * (self.texture_count)
            for i in range(self.texture_count):
                self.texture_ids[i] = self._io.read_s4le()

            self.info = Rsm.NodeInfo(self._io, self, self._root)
            self.node_vertex_count = self._io.read_s4le()
            self.node_vertices = [None] * (self.node_vertex_count)
            for i in range(self.node_vertex_count):
                self.node_vertices[i] = Rsm.NodeVertex(self._io, self,
                                                       self._root)

            self.texture_vertex_count = self._io.read_s4le()
            self.texture_vertices = [None] * (self.texture_vertex_count)
            for i in range(self.texture_vertex_count):
                self.texture_vertices[i] = Rsm.TextureVertex(
                    self._io, self, self._root)

            self.face_count = self._io.read_s4le()
            self.faces_info = [None] * (self.face_count)
            for i in range(self.face_count):
                self.faces_info[i] = Rsm.FaceInfo(self._io, self, self._root)

            if self._root.version >= 261:
                self.pos_key_count = self._io.read_s4le()

            if self._root.version >= 261:
                self.pos_key_frames = [None] * (self.pos_key_count)
                for i in range(self.pos_key_count):
                    self.pos_key_frames[i] = Rsm.PosKeyFrame(
                        self._io, self, self._root)

            self.rot_key_count = self._io.read_s4le()
            self.rot_key_frames = [None] * (self.rot_key_count)
            for i in range(self.rot_key_count):
                self.rot_key_frames[i] = Rsm.RotKeyFrame(
                    self._io, self, self._root)
