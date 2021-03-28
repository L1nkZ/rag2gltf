from typing import List

import glm  # type: ignore

from parsing.rsm import Rsm


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


def decode_string(string: Rsm.String) -> str:
    return _decode_zstr(string.value)


def _decode_zstr(zstr: bytes) -> str:
    return zstr.split(b"\0")[0].decode("cp1252")