import copy
import io
import struct
from typing import List, Tuple, Optional

import glm  # type: ignore

from parsing.rsm import Rsm


def decode_string(string: Rsm.String) -> str:
    return _decode_zstr(string.value)


def _decode_zstr(zstr: bytes) -> str:
    return zstr.split(b"\0")[0].decode("cp1252")


def mat3tomat4(mat3: List[float]) -> glm.mat4:
    mat4 = glm.mat4()
    mat4[0][0] = mat3[0]
    mat4[0][1] = mat3[1]
    mat4[0][2] = mat3[2]
    mat4[1][0] = mat3[3]
    mat4[1][1] = mat3[4]
    mat4[1][2] = mat3[5]
    mat4[2][0] = mat3[6]
    mat4[2][1] = mat3[7]
    mat4[2][2] = mat3[8]
    return mat4


def rag_mat4_mul(mat1: glm.mat4, mat2: glm.mat4) -> glm.mat4:
    matrix1: List[float] = sum(mat1.to_list(), [])
    matrix2: List[float] = sum(mat2.to_list(), [])
    return glm.mat4(
        matrix1[0] * matrix2[0] + matrix1[1] * matrix2[4] +
        matrix1[2] * matrix2[8] + matrix1[3] * matrix2[12],
        matrix1[0] * matrix2[1] + matrix1[1] * matrix2[5] +
        matrix1[2] * matrix2[9] + matrix1[3] * matrix2[13],
        matrix1[0] * matrix2[2] + matrix1[1] * matrix2[6] +
        matrix1[2] * matrix2[10] + matrix1[3] * matrix2[14],
        matrix1[0] * matrix2[3] + matrix1[1] * matrix2[7] +
        matrix1[2] * matrix2[11] + matrix1[3] * matrix2[15],
        matrix1[4] * matrix2[0] + matrix1[5] * matrix2[4] +
        matrix1[6] * matrix2[8] + matrix1[7] * matrix2[12],
        matrix1[4] * matrix2[1] + matrix1[5] * matrix2[5] +
        matrix1[6] * matrix2[9] + matrix1[7] * matrix2[13],
        matrix1[4] * matrix2[2] + matrix1[5] * matrix2[6] +
        matrix1[6] * matrix2[10] + matrix1[7] * matrix2[14],
        matrix1[4] * matrix2[3] + matrix1[5] * matrix2[7] +
        matrix1[6] * matrix2[11] + matrix1[7] * matrix2[15],
        matrix1[8] * matrix2[0] + matrix1[9] * matrix2[4] +
        matrix1[10] * matrix2[8] + matrix1[11] * matrix2[12],
        matrix1[8] * matrix2[1] + matrix1[9] * matrix2[5] +
        matrix1[10] * matrix2[9] + matrix1[11] * matrix2[13],
        matrix1[8] * matrix2[2] + matrix1[9] * matrix2[6] +
        matrix1[10] * matrix2[10] + matrix1[11] * matrix2[14],
        matrix1[8] * matrix2[3] + matrix1[9] * matrix2[7] +
        matrix1[10] * matrix2[11] + matrix1[11] * matrix2[15],
        matrix1[12] * matrix2[0] + matrix1[13] * matrix2[4] +
        matrix1[14] * matrix2[8] + matrix1[15] * matrix2[12],
        matrix1[12] * matrix2[1] + matrix1[13] * matrix2[5] +
        matrix1[14] * matrix2[9] + matrix1[15] * matrix2[13],
        matrix1[12] * matrix2[2] + matrix1[13] * matrix2[6] +
        matrix1[14] * matrix2[10] + matrix1[15] * matrix2[14],
        matrix1[12] * matrix2[3] + matrix1[13] * matrix2[7] +
        matrix1[14] * matrix2[11] + matrix1[15] * matrix2[15])


def decompose_matrix(
    mat: glm.mat4
) -> Tuple[Optional[glm.vec3], Optional[glm.quat], Optional[glm.vec3]]:
    sx = glm.length(glm.vec3(mat[0]))
    sy = glm.length(glm.vec3(mat[1]))
    sz = glm.length(glm.vec3(mat[2]))
    if glm.determinant(mat) < 0.0:
        sx = -sx

    translation = glm.vec3(mat[3])
    scale = glm.vec3(sx, sy, sz)

    inv_sx = 1.0 / sx
    inv_sy = 1.0 / sy
    inv_sz = 1.0 / sz

    rot_mat = copy.copy(mat)
    rot_mat[0][0] *= inv_sx
    rot_mat[0][1] *= inv_sx
    rot_mat[0][2] *= inv_sx
    rot_mat[1][0] *= inv_sy
    rot_mat[1][1] *= inv_sy
    rot_mat[1][2] *= inv_sy
    rot_mat[2][0] *= inv_sz
    rot_mat[2][1] *= inv_sz
    rot_mat[2][2] *= inv_sz
    rot_mat[3] = glm.vec4(0.0, 0.0, 0.0, 1.0)
    rotation = glm.normalize(glm.quat_cast(rot_mat))

    if translation == glm.vec3():
        translation = None
    if rotation == glm.quat():
        rotation = None
    if scale == glm.vec3():
        scale = None

    return (translation, rotation, scale)


def serialize_floats(array: List[float], stream: io.BytesIO) -> int:
    written = 0
    for value in array:
        written += stream.write(struct.pack('f', value))
    return written