import io
from pathlib import Path

import numpy
from PIL import Image as PillowImage  # type: ignore


def convert_bmp_to_png(image_path: Path) -> io.BytesIO:
    image = PillowImage.open(image_path).convert("RGBA")
    image = _remove_magenta_pixels(image)
    return _convert_image_to_png(image)


def convert_tga_to_png(image_path: Path) -> io.BytesIO:
    image = PillowImage.open(image_path)
    return _convert_image_to_png(image)


def _convert_image_to_png(image: PillowImage) -> io.BytesIO:
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="png")
    bytes_io.seek(0)
    return bytes_io


def _remove_magenta_pixels(image: PillowImage) -> PillowImage:
    return _magenta_to_transparent_vectorized(image)


def _magenta_to_transparent_vectorized(image: PillowImage) -> PillowImage:
    arr = numpy.array(image)

    with numpy.errstate(invalid='ignore'):
        # RGB to HSV
        maxc = arr.max(-1)
        minc = arr.min(-1)

        hsv = numpy.zeros(arr.shape)
        hsv[:, :, 2] = maxc
        hsv[:, :, 1] = (maxc - minc) / maxc

        divs = (maxc[..., None] - arr) / ((maxc - minc)[..., None])
        cond1 = divs[..., 0] - divs[..., 1]
        cond2 = 2.0 + divs[..., 2] - divs[..., 0]
        h = 4.0 + divs[..., 1] - divs[..., 2]
        h[arr[..., 2] == maxc] = cond1[arr[..., 2] == maxc]
        h[arr[..., 1] == maxc] = cond2[arr[..., 1] == maxc]
        hsv[:, :, 0] = (h / 6.0) % 1.0

        hsv[minc == maxc, :2] = 0

        # Convert magenta pixels to fully transparent pixels
        c1 = numpy.logical_and(hsv[:, :, 0] > 0.75, hsv[:, :, 0] < 0.90)
        c2 = numpy.logical_and(c1, hsv[:, :, 1] >= 0.4)
        condition = numpy.logical_and(c2, hsv[:, :, 2] >= 0.3)

        arr[:, :, 0] = (~condition) * arr[:, :, 0]
        arr[:, :, 1] = (~condition) * arr[:, :, 1]
        arr[:, :, 2] = (~condition) * arr[:, :, 2]
        arr[:, :, 3] = (~condition) * arr[:, :, 3]

    return PillowImage.fromarray(arr)
