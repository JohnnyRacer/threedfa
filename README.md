# **ThreeDFA**

## No face left unaligned ✊✊✊

### A fast, accurate and easy to use all-in-one solution for facial feature extraction. 
### Simple and streamlined detection and alignment API with mesh capbilities and depth estimation. 

### Code was adapted from cleardusk's 3DDFA_V2 : https://github.com/cleardusk/3DDFA_V2

## Features

- Module and web based API for ease of use.

- 3D facial landmarks via depth estimation and perspective reconstruction.

- Supports inferencing on both ONNX runtime (CPU/GPU) as well as Pytorch.

- Mesh interpolation and export with textured UV map.

- Capable of generating dlib landmarks (68 points), mesh landmarks (35568 points) or **EXPERIMENTAL**  mediapipe/ARKit/Unity landmarks (468 points).

## Installation

Simply run `python -m build` and then install the wheel using `pip install dist/threedfa.XXX.whl`

TODOs : Building manylinux wheel for PyPi support. 

## Requirements

#### Building Cython Extensions

Linux based distribution with gcc ⩾ 8.x or equivalent clang compiler for building cypthon versions of FaceBoxes and Sim3DR. Building on other OSes are untested.  

#### Pip Dependencies
`torch
torchvision
pillow
numpy
opencv-python
imageio
imageio-ffmpeg
pyyaml
tqdm
argparse
cython
scikit-image
scipy
onnxruntime
`

### Hardware

#### **Inferencing on the CPU**

A relative newer x86-64 CPU with decent floating point performance, with AVX preferable if using ONNX runtime on CPU. Most CPUs made after 2014 should give very satisfactory performance, with inference times of between 2-5 seconds for a 2K image. This code base has not been tested at all with ARM based CPUs, so compatibility and speeds are unknown. Please submit a PR if you are willing to share code or results related to ARM CPUs.

#### **Inferencing on the GPU**

By default ONNX will be using the CPU for inference, GPU inference for ONNX can be enabled by uninstalling the `onnxruntime` package from pip and replacing it with `onnxruntime-gpu`. Please do note that the highest supported version of ONNX runtime is 1.8.0 for GPU inference. This is due to how ONNX has changed its providers allocation option parsing internally.

#### **Inferencing on other XLA Accerlators**

Untested. Although Pytorch XLA should be supported.

### Credits

Thanks to **cleardusk** for providing an amazing repo and codebase.

Thanks to **rednafi** for `fastapi-nano`.