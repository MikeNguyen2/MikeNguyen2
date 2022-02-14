import pyrealsense2 as rs
import numpy as np
import rospy
import time
import math
import bot
import cv2
import sys

rospy.init_node('move_group_interface_bot')
cobot = bot.MoveGroupInterfaceBot()
#cobot.move_h(5, 100) # 5 = grab firmly, 4 = press button
#sys.exit()

start_x = 0.045
start_y = 0.25
start_z = 0.3 #0.3
#cobot.move_e(start_x, start_y, start_z, 0, 0, 0)

pipeline = rs.pipeline()
config = rs.config()

wrapper = rs.pipeline_wrapper(pipeline)
profile = config.resolve(wrapper)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

h1 = 0
s1 = 0
v1 = 0
h2 = 1
s2 = 1
v2 = 1

### green mtp ###
mtp = [35, 40, 90, 75, 115, 210]
#h1 = 35
#s1 = 40
#v1 = 90
#h2 = 75
#s2 = 115
#v2 = 210

### red liquid tube ###
tube = [0, 95, 75, 20, 255, 150]
#h1 = 0
#s1 = 95
#v1 = 75
#h2 = 20
#s2 = 255
#v2 = 150

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

time.sleep(2)

def look(h1, s1, v1, h2, s2, v2):
  frames = pipeline.wait_for_frames()
  frames = align.process(frames)

  frame_depth = frames.get_depth_frame()
  frame_color = frames.get_color_frame()
  if not frame_depth or not frame_color:
    return

  img_depth = np.asanyarray(frame_depth.get_data())
  img_color = np.asanyarray(frame_color.get_data())

  map = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.5), cv2.COLORMAP_JET)

  sensor_depth = profile.get_device().first_depth_sensor()
  scale_depth = sensor_depth.get_depth_scale()

  intrinsics = frame_depth.profile.as_video_stream_profile().intrinsics
  extrin_depth_to_color = frame_depth.profile.get_extrinsics_to(frame_color.profile)

  #print('extrin_depth_to_color')
  #print(extrin_depth_to_color)
  #print('')

  img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
  hsv1 = np.asarray([h1, s1, v1])
  hsv2 = np.asarray([h2, s2, v2])
  img_mask = cv2.inRange(img_hsv, hsv1, hsv2)

  n = 0
  avg_X = 0
  avg_Y = 0
  avg_x = 0
  avg_y = 0
  avg_z = 0
  for y in range(img_mask.shape[0]):
    for x in range(img_mask.shape[1]):
      if x <= 0 or x >= 1280 or y <= 0 or y >= 720:
        continue
      if img_mask.item(y, x) > 250:
        distance = frame_depth.get_distance(x, y)
        #print(x, y, distance)
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance)
        n += 1
        avg_X += x
        avg_Y += y
        avg_x += point[0]
        avg_y += point[1]
        avg_z += point[2]

  if n <= 0:
    return 0, 0
  avg_X /= n
  avg_Y /= n
  avg_x /= n
  avg_y /= n
  avg_z /= n

  name = 'Buddy'
  cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(name, img_mask)
  cv2.waitKey(1)

  return avg_x, avg_y

  '''print(avg_X, avg_Y)
  print(avg_x, avg_y, avg_z)
  img_mask = cv2.rectangle(img_mask, (avg_X-1, avg_Y-1), (avg_X+1, avg_Y+1), 128, 4)
  cobot.move_e(avg_x-0.024, -avg_y+0.264, 0.155, -90, -90, 0)
  cobot.move_e(avg_x-0.024, -avg_y+0.264, 0.1225, -90, -90, 0)
  sys.exit()'''


def look_average(h1, s1, v1, h2, s2, v2):
  n = 10
  avg_x = 0
  avg_y = 0

  for i in range(n):
    x, y = look(h1, s1, v1, h2, s2, v2)
    print('look', i, x, y)
    avg_x += x
    avg_y += y

  avg_x /= n
  avg_y /= n
  return avg_x, avg_y


if __name__ == '__main__':
  try:
    def press_button():
      cobot.move_h(4, 100)
      cobot.move_h(5, 100)

    cobot.move_h(5, 100)
    cobot.move_e(0.045, 0.25, 0.3, 0, 0, 0)
    #sys.exit()

    # get tip
    cobot.move_e(-0.1, 0.2, 0.105, -90, -90, 0)
    #sys.exit()
    cobot.move_e(-0.1, 0.2, 0.093, -90, -90, 0)
    cobot.move_e(-0.1, 0.2, 0.105, -90, -90, 0)
    cobot.move_e(-0.1, 0.2, 0.140, -90, -90, 0)
    cobot.move_e(-0.1, 0.2, 0.22, -90, -90, 0)

    # get liquid
    cobot.move_e(0, 0.275, 0.22, -90, -90, 0)
    #sys.exit()
    cobot.move_e(0, 0.275, 0.18, -90, -90, 0)
    press_button()
    time.sleep(1)
    cobot.move_e(0, 0.275, 0.22, -90, -90, 0)

    # pipette
    cobot.move_e(-0.01, 0.225, 0.22, -90, -90, 0)
    cobot.move_e(-0.01, 0.225, 0.125, -90, -90, 0)
    #sys.exit()
    cobot.move_e(-0.01, 0.225, 0.11, -90, -90, 0)
    press_button()
    time.sleep(1)
    cobot.move_e(-0.01, 0.225, 0.125, -90, -90, 0)
    cobot.move_e(-0.01, 0.225, 0.22, -90, -90, 0)

    #print(cobot.current_pose())

    # streife ab
    cobot.move_e(0.0975, 0.15, 0.225, 0, 0, 0)
    cobot.move_e(0.0975, 0.05, 0.13, 0, -20, -90)
    cobot.move_e(0.0975, 0.05, 0.12, 0, -20, -90)
    cobot.move_e(0.0975, 0.05, 0.115, 0, -20, -90)
    cobot.move_e(0.0975, 0.07, 0.1225, 0, -20, -90)
    #sys.exit()

    '''tube_x, tube_y = look_average(0, 95, 75, 20, 255, 150) # get liquid
    mtp_x, mtp_y = look_average(35, 40, 90, 75, 150, 210) # s2 = 115

    x, y = tube_x, tube_y
    cobot.move_e(x+0.018, -y+0.22, 0.25, -90, -90, 0)
    cobot.move_e(x+0.018, -y+0.22, 0.14, -90, -90, 0)
    cobot.move_e(x+0.018, -y+0.22, 0.25, -90, -90, 0)

    x, y = mtp_x, mtp_y
    cobot.move_e(x-0.028, -y+0.25, 0.175, -90, -90, 0) # z = 0.155
    cobot.move_e(x-0.028, -y+0.25, 0.12, -90, -90, 0) # z = 125'''

  except KeyboardInterrupt:
    pipeline.stop()
    sys.exit()
  finally:
    pipeline.stop()
