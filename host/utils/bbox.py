def stabilize_bbox(bbox, bbox_buffer):
    buffer = bbox_buffer
    # Update buffer (FILO)
    if bbox not in buffer:
        buffer.pop(0)
        buffer.append(bbox)
    if not bbox:
        return (), buffer
    
    # Save box center
    cx, cy = bbox[0]+(bbox[2]//2), bbox[1]+(bbox[3]//2)
    ws,hs = [],[]
    for b in buffer:
        if b:
            ws.append(b[2])
            hs.append(b[3])
    if not ws or not hs:
        return (), buffer
    
    # Calc median size
    ws.sort()
    hs.sort()
    w = ws[len(ws)//2]
    h = hs[len(ws)//2]
    # Increase size with 20%
    w = int(w*1.2)
    h = int(h*1.2)
    x = cx - (w//2)
    y = cy - (h//2)
    return (x,y,w,h), buffer

def get_single_bbox(det_dict, class_id, frame_width, frame_height):
    for i in range(det_dict['num_detections']):
        if det_dict['detection_classes'][i] == class_id:
            bbox = format_bbox(det_dict['detection_boxes'][i], frame_width, frame_height) 
            return bbox, int(det_dict['detection_scores'][i] * 100)
    return (), 0

def format_bbox(bbox_norm, frame_width, frame_height):
    ymin, xmin, ymax, xmax = bbox_norm
    x = int(xmin*frame_width)
    y = int(ymin*frame_height)
    w = int(xmax*frame_width) - int(xmin*frame_width)
    h = int(ymax*frame_height) - int(ymin*frame_height)
    return (x,y,w,h) 
