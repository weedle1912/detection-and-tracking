def bbox_stabilize(bbox, bbox_buffer):
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
    #w = int(w*1.2)
    #h = int(h*1.2)
    x = cx - (w//2)
    y = cy - (h//2)
    return (x,y,w,h), buffer

def get_single_bbox(det_dict, class_id, frame_width, frame_height):
    for i in range(det_dict['num_detections']):
        if det_dict['detection_classes'][i] == class_id:
            ymin, xmin, ymax, xmax = det_dict['detection_boxes'][i]
            bbox_n = bbox_format_xywh(xmin,ymin,xmax,ymax)
            bbox = bbox_scale(bbox_n, frame_width, frame_height)
            return bbox, int(det_dict['detection_scores'][i] * 100)
    return (), 0

def bbox_format_xywh(x1,y1,x2,y2):
    return (x1,y1,x2-x1,y2-y1)

def bbox_format_x1y1x2y2(x,y,w,h):
    return (x,y,x+w,y+h)

def bbox_normalize(bbox, frame_width, frame_height, precision):
    if not bbox:
        return ()
    x = round( float(bbox[0])/frame_width,  precision )
    y = round( float(bbox[1])/frame_height, precision )
    w = round( float(bbox[2])/frame_width,  precision )
    h = round( float(bbox[3])/frame_height, precision )
    return (x,y,w,h)

def bbox_scale(bbox_norm, frame_width, frame_height):
    if not bbox_norm:
        return ()
    x = int(bbox_norm[0]*frame_width)
    y = int(bbox_norm[1]*frame_height)
    w = int(bbox_norm[2]*frame_width)
    h = int(bbox_norm[3]*frame_height)
    return (x,y,w,h)
