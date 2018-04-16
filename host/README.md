## Host: PC with Linux
### 1. Description
#### Purpose:
Deep learning and data processing.

#### Requirements:
* Ubuntu 16.04
* TensorFlow 1.7
* TensorRT 3.0

### 2. Setup
#### 2.1 Install TensorRT 3.0
1. Go to: https://developer.nvidia.com/nvidia-tensorrt-download  
(Requires [NVIDIA Developer Program](https://developer.nvidia.com/developer-program) membership)
2. Download the debian install package for TensorRT 3.0 for CUDA 9.0
3. Install TensorRT from the debian package:  
```
$ sudo dpkg -i nv-tensorrt-repo-ubuntu1604-cuda9.0-rc-trt4.x.x.x-yyyymmdd_1-1_amd64.deb  
$ sudo apt-get update  
$ sudo apt-get install tensorrt
```
4. Install Python 2.7 Nvidia infer library:
```
$ sudo apt-get install python-libnvinfer-doc swig
```
5. Install the TF to UFF converter:
```
$ sudo apt-get install uff-converter-tf
```
