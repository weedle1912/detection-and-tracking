# Detection and Tracking
Repository for Master's Thesis in Engineering Cybernetics at NTNU, 2018.

Title: _Autonomous Target Detection and Tracking for Remotely operated Weapon Stations_

#### Intention
Detect and track targets of interest in camera video, for a Remote Weapon Station (RWS).

#### Approach
Combine an accurate detector with a fast tracker.
Methods of interest:
* Detector based on _deep learning_
* _Point based_ tracker

#### Motivation/Design
If a detector process frames slower than the video frame rate, a number of frames will be skipped in between detections - and the detection will also be a few frames "old" when presented.
A solution to this is to buffer the skipped frames, and retrace this buffer with a fast enough tracker, to make the detection relevant for the current frame.    

The implementation `host/tracking-app.py`is an _autonomous tracker_ - a real-time tracker with periodic corrections from a deep learning detector.

#### Software Framework
- TensorFlow: Object Detection API
- OpenCV: contrib package tracking API

#### Hardware
- Host: Desktop PC (optional CPU/GPU depending on TensorFlow distribution)
- Target: Nvidia Jetson TX (future work)
