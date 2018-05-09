import cv2

BGR = {
    'black':(0,0,0),
    'blue':(255,0,0),
    'green':(0,255,0), 
    'orange':(0,153,255),
    'red':(0,0,255),
    'white':(255,255,255)
}

def draw_bbox(frame, bbox, label, color):
    if bbox:
        cv2.rectangle(frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),BGR[color],2)
        draw_label(frame, bbox, label, color)

def draw_label(img, bbox, label, color):
    # Calc position
    s, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1) 
    xb, yb, wb, hb = bbox[0]-1, bbox[1]-s[1]-6, s[0]+2, s[1]+6
    xl, yl = bbox[0], bbox[1]-3
    if yl < 13:
        yb += (bbox[3]+hb)
        yl += (bbox[3]+hb)

    # Draw background and text
    cv2.rectangle(img,(xb,yb),(xb+wb,yb+hb),BGR[color],-1) 
    cv2.putText(img,label,(xl,yl), cv2.FONT_HERSHEY_PLAIN, 1,BGR['white'],1,cv2.LINE_AA)

def draw_header(img, class_name, score):
    cv2.putText(img,'Target: %s'%class_name.capitalize(),(10,20), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)
    cv2.putText(img,( 'Score: ' + ('%d%%'%score).rjust(4) ),(10,35), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)

def draw_footer(img, fps_d, fps_t, no_track, frame_height): 
    cv2.putText(img,( 'Det. FPS: ' + ('%d'%fps_d).rjust(3) ),(10,frame_height-25), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)
    if no_track:
        cv2.putText(img,( 'No track.' ),(10,frame_height-10), cv2.FONT_HERSHEY_PLAIN, 1,BGR['red'],1,cv2.LINE_AA)
    else:
        cv2.putText(img,( 'Trc. FPS: ' + ('%d'%fps_t).rjust(3) ),(10,frame_height-10), cv2.FONT_HERSHEY_PLAIN, 1,BGR['black'],1,cv2.LINE_AA)
