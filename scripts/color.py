import pyrealsense2 as rs
import numpy as np
import cv2 as cv

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

# config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

pipeline.start(config)

mtp_weiss = [200,200,200,255,255,255,'mtp_weiss']
hsv = [10,77,10,69,111,188]
bgr = [11,199,194,221,120,231]

def empty(a):
    pass 

#bgr nicht hsv
cv.namedWindow('trackbar')
cv.resizeWindow('trackbar',640,240)
cv.createTrackbar('hue min','trackbar',0,255,empty)
cv.createTrackbar('hue max','trackbar',255,255,empty)
cv.createTrackbar('sat min','trackbar',0,255,empty)
cv.createTrackbar('sat max','trackbar',255,255,empty)
cv.createTrackbar('val min','trackbar',0,255,empty)
cv.createTrackbar('val max','trackbar',255,255,empty)

#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
v_max =255
try:
    while 0< v_max:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        img = np.asanyarray(color_frame.get_data())
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        adap = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
        h_min = cv.getTrackbarPos('hue min','trackbar')
        h_max = cv.getTrackbarPos('hue max','trackbar')
        s_min = cv.getTrackbarPos('sat min','trackbar')
        s_max = cv.getTrackbarPos('sat max','trackbar')
        v_min = cv.getTrackbarPos('val min','trackbar')
        v_max = cv.getTrackbarPos('val max','trackbar')
        lower = np.array([h_min,s_min,v_min])
        upper = np.array([h_max,s_max,v_max])
        mask = cv.inRange(img,lower,upper)

        cv.imshow('adap',adap)
        cv.imshow('mask',mask)
        cv.waitKey(1)

finally:
    pipeline.stop()