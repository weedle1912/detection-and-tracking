## Collection of practical scripts for testing and development
#### Setup
  - `env_setup.sh`:    
    - Create virtual environment ~/tensorflow
    - Install requirements
    - Download TF Object Detection API

#### Data processing
  - `images_to_video.py`:    
    - Create video from images in specified folder    

  - `make_gt.py`:
    - Make ground truth bbox for object, frame by frame, in specified video

  - `normalize_gt.py`:
    - Normalize ground truth file with rights to specified size

  - `show_bboxes.py`:
    - Play or step through video with specified bbox file

#### Metrics
  - `iou_by_csv.py`: 
    - Calculate the "intersection over union"/overlap for bboxes specified by two .csv-files
  
  - `dist_score_by_csv.py`:
    - Calculate a distence score for bboxes specified by two .csv-files
    - Measure of prediction center relative to ground truth center: 1 for overlap to 0 at ground truth edge
    - Formula: 1 - min( max(2 * error_x_dir / w_G, 2 * error_y_dir / w_G), 1)
