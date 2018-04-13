# ** IMPORT
import numpy as np 
import os
import six.moves.urllib as urllib
import sys
import cv2
import tensorflow as tf 
import time

from collections import defaultdict
from io import StringIO

from object_detection.utils import ops as utils_ops

if tf.__version__ < '1.4.0':
    raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')

# * TensorFlow Object Detection API
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util 

# -----------------------------------------------------------------

# * Load a (frozen) TensorFlow model into memory
def load_tf_graph(model_name):
    cwd_path = os.getcwd()
    path_to_ckpt = os.path.join(cwd_path, 'models', model_name, 'frozen_inference_graph.pb')
    
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# * Load label map
def get_label_index(label_name, num_classes):
    cwd_path = os.getcwd()
    path_to_labels = os.path.join(cwd_path, 'object_detection', 'data', label_name + '.pbtxt')

    label_map = label_map_util.load_labelmap(path_to_labels)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, 
        max_num_classes=num_classes, 
        use_display_name=True
    )
    category_index = label_map_util.create_category_index(categories)
    return category_index

# ** DETECTION
def run_detection(vidcap, detection_graph, category_index):
    with detection_graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = get_handles()
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            while( vidcap.isOpened() ):
                # Get frame
                succsess, image_np = vidcap.read()
                if not succsess:
                    break

                # Resize
                img_h, img_w = image_np.shape[:2]
                img_w = int(img_w*(480.0/img_h))
                img_h = 480
                image_np = cv2.resize(image_np,(img_w, img_h), interpolation = cv2.INTER_AREA)

                # Expand dimension. Model expects shape: [1, None, None, 3]
                image_expanded = np.expand_dims(image_np, axis=0)
                # Run inference
                t = time.time()
                output_dict = sess.run(
                    tensor_dict,
                    feed_dict={image_tensor: image_expanded}
                )    
                # Display FPS in image
                cv2.putText(
                    image_np,
                    ("FPS: %d" % (1/(time.time()-t))),
                    (10, img_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0,0,255), #Red
                    2
                )

                # All outputs are float32 numpy arrays, so convert types to appropriate
                output_dict = convert_appropriate(output_dict)
                # Add bounding boxes to frame
                draw_detections(image_np, output_dict, category_index)

                # Display
                cv2.imshow('frame', image_np)
                if ( cv2.waitKey(1) & 0xFF == ord('q') ):
                    break

def get_handles():
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
    
    if 'detection_masks' in tensor_dict:
        # The following process is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe required. Translate mask from box coordinates to image coordinates, and fit the image size
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1]
        )
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8
        )
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0
        )

    return tensor_dict

def convert_appropriate(output_dict):
    output_dict['num_detections']       = int(output_dict['num_detections'][0])
    output_dict['detection_classes']    = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes']      = output_dict['detection_boxes'][0]
    output_dict['detection_scores']     = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict  

# ** VISUALIZATION
def draw_detections(image_np, detections_dict, category_index):
    # Visualization of the results of a detection
    image_det = image_np
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_det,
        detections_dict['detection_boxes'],
        detections_dict['detection_classes'],
        detections_dict['detection_scores'],
        category_index,
        instance_masks=detections_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=2
    )
    return image_det

    
