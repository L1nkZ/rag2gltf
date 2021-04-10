#!/usr/bin/python3

import multiprocessing

from rag2gltf import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
