import glm  # type: ignore


def mat3tomat4(mat3: glm.mat3) -> glm.mat4:
    mat4 = glm.mat4(1.0)
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


def decode_zstr(zstr: bytes) -> str:
    return zstr.split(b"\0")[0].decode("cp1252")