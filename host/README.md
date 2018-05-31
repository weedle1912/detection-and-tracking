## Host: PC with Linux
### 1. Description
#### Purpose:
Deep learning and data processing.

#### Requirements:
* Ubuntu 16.04
* TensorFlow 1.7
* TensorRT 3.0

### 2. Setup
#### 2.1 Download TensorFlow object_detection API
The TensorFlow object_detection is available on GitHub, and can be downloaded from:    
https://github.com/tensorflow/models/tree/master/research/object_detection    

Store the API in a folder 'object_detection' in 'src/detector/'

#### 2.2 Install the OpenCV-contrib package, for tracking API
The contrib package can be installed with pip:    
```
$ pip install opencv-contrib-python
```  