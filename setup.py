#!/usr/bin/python3

from setuptools import setup, find_packages

setup(name='rag2gltf',
      version='0.1.0',
      description='RSM to glTF 2.0 model converter',
      author='LinkZ',
      author_email='wanthost@gmail.com',
      packages=find_packages(),
      entry_points={'console_scripts': ['rag2gltf=rag2gltf:main']})
