import numpy as np
import cv2

import detect

MODEL_NAME = 'ssd_inception_v2_coco_2017_11_17'
LABEL_NAME = 'mscoco_label_map'
NUM_CLASSES = 90
VIDEO_FILE = '/home/vetlebg/Videos/BlueAngels.mp4'

def main():
    # Load model graph
    print 'Loading frozen model: ' + MODEL_NAME
    detection_graph = detect.load_tf_graph(MODEL_NAME)
    category_index = detect.get_label_index(LABEL_NAME, NUM_CLASSES)

    # Video source
    print 'Video source: ' + VIDEO_FILE
    cap = cv2.VideoCapture(VIDEO_FILE)

    # Process frame-by-frame
    print 'Running detection ...'
    while( cap.isOpened() ):
        suc, frame = cap.read()
        #frame = cv2.resize(frame, (640, 480))

        # Expand dimension. Model expects shape: [1, None, None, 3]
        image_expanded = np.expand_dims(frame, axis=0)

        # Detection
        output_dict = detect.run_inference_for_single_image(frame, detection_graph)
        # Add bounding boxes to frame
        frame_detections = detect.draw_detections(frame, output_dict, category_index)

        # Display
        cv2.imshow('frame', frame_detections)
        if ( cv2.waitKey(1) & 0xFF == ord('q') ):
            break

    cap.release()
    cv2.destroyAllWindows()






if __name__ == '__main__':
    main()