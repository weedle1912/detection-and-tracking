## Target: NVIDIA Jetson TX2 Development Kit
### 1 Description
**Purpose**:  
System deployment.

**Requirements**:
* CUDA Toolkit 9.0
* cuDNN 7.0
* TensorRT 3.0
* OpenCV 3.3.1

### 2 Optimize performance on the TX2
#### 2.1 Max frequency to CPU, GPU and EMC clocks
Maximize jetson performance by setting static max frequency to CPU, GPU and EMC clocks, with the `~/jetson_clocks.sh` script:
1. Set max:  
1.1 Store current settings (\[file\] is optional, default is {$HOME}/l4t_dfs.conf):  
`$ ~/jetson_clocks.sh --store [file]`  
1.2 Set static max frequency:  
`$ ~/jetson_clocks.sh`  

2. Restore previously stored settings:  
`$ ~/jetson_clocks.sh --restore [file]`

#### 2.2 Specify different CPU and GPU modes
With the command line tool `nvpmodel`, the TX2 can use different power modes for CPU and GPU frequencies, as well as CPU core activation:  
`$ sudo nvpmodel -m [mode]`  

The following models are predefined (CPUs ARM A57 quad core and Denver dual core):  

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
