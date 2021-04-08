# rag2gltf

Command-line utility to convert RSM1 and RSM2 models to glTF 2.0.

## Prerequisites

* Python >= 3.7

## How To

```
$ git clone https://github.com/L1nkZ/rag2gltf.git
$ cd rag2gtlf
$ pip install -r requirements.txt
$ python rag2gltf data/model/gld2/building.rsm --data_folder=data/
```

## Limitations

* Texture animations (RSM>=2.3) aren't supported by glTF 2.0
