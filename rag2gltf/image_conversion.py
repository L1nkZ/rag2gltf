import colorsys
import io
from pathlib import Path

from PIL import Image as PillowImage  # type: ignore


def convert_bmp_to_png(image_path: Path) -> io.BytesIO:
    image = PillowImage.open(image_path).convert("RGBA")
    _remove_magenta_pixels(image)
    return _convert_image_to_png(image)


def convert_tga_to_png(image_path: Path) -> io.BytesIO:
    image = PillowImage.open(image_path)
    return _convert_image_to_png(image)


def _convert_image_to_png(image: PillowImage) -> io.BytesIO:
    bytes_io = io.BytesIO()
    image.save(bytes_io, format="png")
    bytes_io.seek(0)
    return bytes_io


def _remove_magenta_pixels(image: PillowImage) -> None:
    _magenta_to_transparent(image)


def _magenta_to_transparent(image: PillowImage) -> None:
    for y in range(image.height):
        for x in range(image.width):
            pixel_color = image.getpixel((x, y))
            r = pixel_color[0]
            g = pixel_color[1]
            b = pixel_color[2]
            hsv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if hsv[0] > 0.75 and hsv[0] < 0.90 and hsv[1] >= 0.4 and hsv[
                    2] >= 0.3:
                image.putpixel((x, y), (0, 0, 0, 0))
