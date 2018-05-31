## Host: PC with Linux
### 1. Description
#### Purpose:
Object tracking application running on host computer.

#### Requirements:
* Ubuntu 16.04
* TensorFlow 1.8
* Protobuf 3.0
* Google's object_detection API
* OpenCV-contrib for tracking API

### 2. Setup
#### 2.1 Install the newest version of Protobuf    
```
$ curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip
$ unzip protoc-3.2.0-linux-x86_64.zip -d protoc3
$ sudo mv protoc3/bin/* /usr/local/bin/
$ sudo mv protoc3/include/* /usr/local/include/
$ rm -r protoc*
```
### 2.2 Install virtual environment    
```
$ sudo apt-get install -y python-pip python-dev python-virtualenv
```
### 2.3 Run the setup script    
```
$ chmod +x ./scripts/env_setup.sh
$ ./scripts/env_setup.sh
```

### 2.4 Compile Protobuf libraries
```
$ cd src/detector
$ sudo protoc object_detection/protos/*.proto --python_out=.
$ cd ../..
```

### 3. Run
Activate the virtual environment, and run application:    
```
$ source ~/tensorflow/bin/activate
(tensorflow)$ python tracking-app.py
```