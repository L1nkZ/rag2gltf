import fire  # type: ignore

from .rsm_conversion import convert_rsm


def main():
    fire.Fire(convert_rsm)