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
cobot.move_h(4, 100)
#sys.exit()

start_x = 0.045
start_y = 0.25
start_z = 0.3
cobot.move_e(start_x, start_y, start_z, 0, 0, 0)

pipeline = rs.pipeline()
config = rs.config()

wrapper = rs.pipeline_wrapper(pipeline)
profile = config.resolve(wrapper)

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

time.sleep(2)

def look():
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
  img_show = img_color.copy()

  img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)
  hsv1 = np.asarray([35, 52, 90])
  hsv2 = np.asarray([75, 115, 210])
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
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], distance)
        n += 1
        avg_X += x
        avg_Y += y
        avg_x += point[0]
        avg_y += point[1]
        avg_z += point[2]

  if n <= 0:
    return
  avg_X /= n
  avg_Y /= n
  avg_x /= n
  avg_y /= n
  avg_z /= n
  print(avg_X, avg_Y)
  print(avg_x, avg_y, avg_z)
  img_show = cv2.rectangle(img_show, (avg_X-1, avg_Y-1), (avg_X+1, avg_Y+1), 128, 4)
  #cobot.move_e(avg_x-0.026, -avg_y+0.2525, 0.155, -90, -90, 0)
  #cobot.move_e(avg_x-0.026, -avg_y+0.2525, 0.125, -90, -90, 0)
  #sys.exit()


  ### BEGIN MTP DETECTION WITH ROTATION ###
  img_mask = cv2.GaussianBlur(img_mask, (5, 5), 0)
  bx, by, bw, bh = cv2.boundingRect(img_mask)

  height, width = img_mask.shape
  img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

  minx = width
  miny = height
  min = (width * height) ** 2
  for x in range(0, width):
    for y in range(0, height):
      if img_mask.item(y, x) > 128:
        if y*y+x*x < min:
          min = y*y+x*x
          miny = y
          minx = x

  x, y = minx, miny
  m = 8

  contours = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
  img_mask = cv2.drawContours(img_show, contours, 0, (255, 128, 0), 4)

  if (len(contours) == 0):
    return
  contour = contours[0]

  cv2.rectangle(img_show, (bx+1, by+1), (bx+bw, by+bh), (0, 0, 0), 2)
  cv2.rectangle(img_show, (bx, by), (bx+bw, by+bh), (50, 50, 200), 2)

  font = cv2.FONT_HERSHEY_SIMPLEX
  def drawText(text, x, y, r, g, b):
    cv2.putText(img_show, text,
      (x+1, y+1), font, 0.75, (0, 0, 0), 2, cv2.LINE_AA
    )
    cv2.putText(img_show, text,
      (x, y), font, 0.75, (r, g, b), 2, cv2.LINE_AA
    )

  drawText('MTP', bx+25, by-95, 50, 50, 200)
  drawText('x = ' + str(bx), bx+25, by-60, 50, 50, 200)
  drawText('y = ' + str(by), bx+25, by-25, 50, 50, 200)

  peri = cv2.arcLength(contour, True)
  approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
  print(len(approx))
  if (len(approx) != 5):
    return

  for point in approx:
    x, y = point[0]
    cv2.line(img_show, (x-m+1, y-m+1), (x+m+1, y+m+1), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img_show, (x+m+1, y-m+1), (x-m+1, y+m+1), (0, 0, 0), 2, cv2.LINE_AA)
    cv2.line(img_show, (x-m, y-m), (x+m, y+m), (0, 255, 0), 2, cv2.LINE_AA)
    cv2.line(img_show, (x+m, y-m), (x-m, y+m), (0, 255, 0), 2, cv2.LINE_AA)


  closest = []
  shortestSq = (1920*1080)**2
  for p1 in approx:
    p1 = p1[0]
    for p2 in approx:
      p2 = p2[0]
      if p1[0] == p2[0] and p1[1] == p2[1]:
        continue
      distanceSq = (p2[0]-p1[0])*(p2[0]-p1[0])+(p2[1]-p1[1])*(p2[1]-p1[1])
      if distanceSq < shortestSq:
        shortestSq = distanceSq
        closest = [p1, p2]


  cornerX = (closest[0][0]+closest[1][0])/2
  cornerY = (closest[0][1]+closest[1][1])/2

  top_left_point = approx[0][0]
  furtherstSq = 0
  for p in approx:
    p = p[0]
    if p[0] == closest[0][0] or p[1] == closest[0][1]:
      continue
    if p[0] == closest[1][0] or p[1] == closest[1][1]:
      continue
    #print(cornerX, cornerY, p[1])
    distanceSq = (p[0]-cornerX)*(p[0]-cornerX)+(p[1]-cornerY)*(p[1]-cornerY)
    if distanceSq > furtherstSq:
      furtherstSq = distanceSq
      top_left_point = p

  drawText('Top left', top_left_point[0]-125, top_left_point[1], 0, 255, 0)

  degrees = math.degrees(math.atan2(cornerX-top_left_point[0], cornerY-top_left_point[1])) - 45
  drawText(str(int(degrees-12)) + 'deg', top_left_point[0]+25, top_left_point[1]+25, 0, 255, 0)

  ### END MTP DETECTION WITH ROTATION ###


  name = 'Buddy'
  cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
  cv2.imshow(name, img_show)
  cv2.waitKey(1)


if __name__ == '__main__':
  try:
    while True:
      look()
  except KeyboardInterrupt:
    pipeline.stop()
    sys.exit()
  finally:
    pipeline.stop()
