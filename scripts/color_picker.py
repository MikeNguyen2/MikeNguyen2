import pyrealsense2 as rs
import numpy as np
import time
import math
import cv2
import sys

h1 = 0
s1 = 0
v1 = 0
h2 = 1
s2 = 1
v2 = 1

h1 = 135
s1 = 60
v1 = 35
h2 = 195
s2 = 169
v2 = 76

cv2.namedWindow('Controls')

def on_change_h1(value):
  global h1
  h1 = value
slider_h1 = cv2.createTrackbar('h1', 'Controls', h1, 255, on_change_h1)

def on_change_s1(value):
  global s1
  s1 = value
slider_s1 = cv2.createTrackbar('s1', 'Controls', s1, 255, on_change_s1)

def on_change_v1(value):
  global v1
  v1 = value
slider_v1 = cv2.createTrackbar('v1', 'Controls', v1, 255, on_change_v1)

def on_change_h2(value):
  global h2
  h2 = value
slider_h2 = cv2.createTrackbar('h2', 'Controls', h2, 255, on_change_h2)

def on_change_s2(value):
  global s2
  s2 = value
slider_s2 = cv2.createTrackbar('s2', 'Controls', s2, 255, on_change_s2)

def on_change_v2(value):
  global v2
  v2 = value
slider_v2 = cv2.createTrackbar('v2', 'Controls', v2, 255, on_change_v2)

pipeline = rs.pipeline()
config = rs.config()

wrapper = rs.pipeline_wrapper(pipeline)
profile = config.resolve(wrapper)

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype('uint8')
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out


def look():
  frames = pipeline.wait_for_frames()
  frames = align.process(frames)
  frame_color = frames.get_color_frame()
  img_color = np.asanyarray(frame_color.get_data())

  intrinsics = frame_color.profile.as_video_stream_profile().intrinsics
  img_show = img_color.copy()

  img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
  hsv1 = np.asarray([h1, s1, v1])
  hsv2 = np.asarray([h2, s2, v2])
  img_mask = cv2.inRange(img_hsv, hsv1, hsv2)

  img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

  bx, by, bw, bh = cv2.boundingRect(img_mask)
  img_mask_rgb = cv2.rectangle(img_mask_rgb, (bx, by), (bx+bw,by+bh), (255, 128, 0), 2)

  name = 'Buddy'
  cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(name, img_mask_rgb)
  cv2.waitKey(1)


if __name__ == '__main__':
  while True:
    look()
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
