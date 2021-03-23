#!/usr/bin/python3
import fire  # type: ignore

from rsm_conversion import convert_rsm

if __name__ == "__main__":
    fire.Fire(convert_rsm)
