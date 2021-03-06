## Target: NVIDIA Jetson TX2 Development Kit
### 1. Description
#### Purpose:  
System deployment.

#### Requirements:
(These should be installed when flashing the TX2 with JetPack 3.2)  
* CUDA Toolkit 9.0
* cuDNN 7.0
* TensorRT 3.0
* OpenCV 3.3.1

### 2. Setup
#### 2.1 Flash TX2 with JetPack 3.2
Include the following installations:
- CUDA Toolkit 90
- cuDNN 7.0
- TensorRT 3.0
- OpenCV 3.3.1

#### 2.2 Build and install OpenCV from source
1. Remove old OpenCV installations and install necessary dependencies:   
`$ ./scripts/OpenCV_prep.sh`
2. Edit the file `/usr/local/cuda/include/cuda_gl_interop.h` to patch OpenGL related compilation problems.
This is done by commenting out the lines #62-66 and #68. Lines #62-68 should look like:  
```c
//#if defined(__arm__) || defined(__aarch64__)
//#ifndef GL_VERSION
//#error Please include the appropriate gl headers before including cuda_gl_interop.h
//#endif
//#else
#include <GL/gl.h>
//#endif
```
And fix symbolic link:   
```
$ cd /usr/lib/aarch64-linux-gnu/
$ sudo ln -sf tegra/libGL.so libGL.so
```

3. Download source and create Makefile for OpenCV:   
`$ ./scripts/OpenCV_init_build.sh`

### 3. Optimize performance on the TX2
#### 3.1 Max frequency to CPU, GPU and EMC clocks
Maximize jetson performance by setting static max frequency to CPU, GPU and EMC clocks, with the `~/jetson_clocks.sh` script:
1. Set max:  
1.1 Store current settings (\[file\] is optional, default is `{$HOME}/l4t_dfs.conf`):  
`$ ~/jetson_clocks.sh --store [file]`  
1.2 Set static max frequency:  
`$ ~/jetson_clocks.sh`  

2. Restore previously stored settings:  
`$ ~/jetson_clocks.sh --restore [file]`

#### 3.2 Specify different CPU and GPU modes
With the command line tool `nvpmodel`, the TX2 can use different power modes for CPU and GPU frequencies, as well as CPU core activation:  
`$ sudo nvpmodel -m [mode]`  

The following models are predefined (quad core ARM A57 and dual core Denver):  

| Mode | Mode name | Denver | Frequency | Arm A57 | Frequency | GPU Frequency |  
| --- | --- | --- | --- | --- | --- | --- |   
| 0	| Max-N	| 2	| 2.0 GHz	| 4	| 2.0 GHz	| 1.30 Ghz |
| 1	| Max-Q	| 0	|         | 4 | 1.2 Ghz	| 0.85 Ghz |
| 2	| Max-P Core-All | 2 | 1.4 GHz | 4 | 1.4 GHz | 1.12 Ghz |
| 3	| Max-P ARM	     | 0 |         | 4 | 2.0 GHz | 1.12 Ghz |
| 4	| Max-P Denver   | 2 | 2.0 GHz | 0 |         | 1.12 Ghz |

These modes are defined in `/etc/nvpmodel.conf`, which can be updated with custom mode definitions.  
Check current mode with:  
`$ sudo nvpmodel -q --verbose`
