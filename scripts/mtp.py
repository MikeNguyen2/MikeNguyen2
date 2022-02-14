import bot
import traceback
import sys
import rospy
import tf2_ros
from sensor_msgs.msg import PointCloud2, Image
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs import point_cloud2
import struct
import ctypes
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import math

rospy.init_node('move_group_interface_bot')
cobot = bot.MoveGroupInterfaceBot()
cobot.move_h(0, 100)
cobot.move_e(0.045, 0.2, 0.2, 0, 0, 0)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

first_callback = True

def cell_offset(c='A', n=1, gap_size=0.009):
  x_index = n - 1
  y_index = ord(c.lower()) - 97

  return x_index * gap_size, y_index * gap_size


def pointcloud_callback(point_cloud):
      transform = tf_buffer.lookup_transform(
        'cam_d435i', 'base_link', rospy.Time(0)
      )
      transformed_point_cloud = do_transform_cloud(point_cloud, transform)
      points = point_cloud2.read_points(transformed_point_cloud)
      n = 0
      avg_x = 0
      avg_y = 0
      avg_z = 0
      avg_r = 0
      avg_g = 0
      avg_b = 0
      for point in points:
        x = point[0]
        y = point[1]
        z = point[2]

        #extract rgb from fourth value (nan)
        #rgb = point[3]
        #s = struct.pack('>f', rgb)
        #i = struct.unpack('>l', s)[0]
        #pack = ctypes.c_uint32(i).value
        #r = (pack & 0x00FF0000)>> 16
        #g = (pack & 0x0000FF00)>> 8
        #b = (pack & 0x000000FF)

        #if r < 100:
          #avg_r += r
          #avg_g += g
          #avg_b += b

        #if abs(x) > 0.1:
        #  continue
        if z > -0.07 and z < -0.04:
          avg_x += x
          avg_y += y
          avg_z += z
          n += 1

      if n < 1:
        return
      avg_x /= n
      avg_y /= n
      avg_z /= n
      #avg_r /= n
      #avg_g /= n
      #avg_b /= n

      print(n)
      print(avg_x, avg_y, avg_z)
      print(avg_r, avg_g, avg_b)
      global first_callback
      if first_callback:
        xoff, yoff = cell_offset('A', 1, 0.009)
        cobot.move_e(avg_x-0.065+xoff, avg_y+0.535-yoff, 0.0475, 0, 0, 0)
        first_callback = False


#rospy.Subscriber('/camera/depth_registered/points', PointCloud2, pointcloud_callback)
#rospy.spin()

#thresh = 160
bridge = CvBridge()

cv2.namedWindow('Controls')

h1 = 0
s1 = 0
v1 = 0
h2 = 255
s2 = 255
v2 = 255

h1 = 35 # 40
s1 = 52 # 55
v1 = 90
h2 = 75
s2 = 115
v2 = 210

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

def fillhole(input_image):
    im_flood_fill = input_image.copy()
    h, w = input_image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    im_flood_fill = im_flood_fill.astype("uint8")
    cv2.floodFill(im_flood_fill, mask, (0, 0), 255)
    im_flood_fill_inv = cv2.bitwise_not(im_flood_fill)
    img_out = input_image | im_flood_fill_inv
    return img_out


def find_corners(img):
  dst = cv2.cornerHarris(img, 5, 3, 0.04)
  ret, dst = cv2.threshold(dst, 0.1*dst.max(), 255, 0)
  dst = np.uint8(dst)
  ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
  corners = cv2.cornerSubPix(img,np.float32(centroids),(5,5),(-1,-1),criteria)
  for i in range(1, len(corners)):
    print(corners[i])
  #img[dst>0.1*dst.max()]=[0,0,255]
  return img


def image_callback(data):
  try:
    #global thresh
    global h1, s1, v1, h2, s2, v2

    img_raw = bridge.imgmsg_to_cv2(data, 'bgr8')
    img = img_raw.copy()
    #img[:,:,0] = 0 # set red and blue channel
    #img[:,:,2] = 0 # to zero, isolating green

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv1 = np.asarray([h1, s1, v1])
    hsv2 = np.asarray([h2, s2, v2])
    img_mask = cv2.inRange(img_hsv, hsv1, hsv2)

    img_mask = cv2.GaussianBlur(img_mask, (5, 5), 0)


    bx, by, bw, bh = cv2.boundingRect(img_mask)

    scale = 2
    height, width = img_mask.shape
    img_small = cv2.resize(img_mask, (width/scale, height/scale))
    img_mask_rgb = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2RGB)

    minx = width
    miny = height
    min = (width * height) ** 2
    for i in range(0, width / scale):
      for j in range(0, height / scale):
        if img_small.item(j, i) > 128:
          y, x = j * scale, i * scale
          if y*y+x*x < min:
            min = y*y+x*x
            miny = y
            minx = x

    '''for i in range(0, width / scale):
      for j in range(0, height / scale):
        if img_small.item(j, i) > 128:
          y, x = j * scale, i * scale
          cv2.rectangle(img_mask_rgb, (x, y), (x+1, y+1), (200, 50, 50), 2)'''


    x, y = minx, miny
    m = 16

    #cv2.rectangle(img_mask_rgb, (bx, by), (bx+bw, by+bh), (50, 200, 50), 2)

    #cv2.rectangle(img_mask_rgb, (x-m, y), (x+m, y), (200, 50, 50), 2)
    #cv2.rectangle(img_mask_rgb, (x, y-m), (x, y+m), (200, 50, 50), 2)

    #cv2.putText(img_mask_rgb, 'x = ' + str(bx) + ", y = " + str(by),
    #  (bx+25, by-25), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 50, 50), 2
    #)

    contours = cv2.findContours(img_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
    img_mask = cv2.drawContours(img_mask_rgb, contours, 0, (255, 128, 0), 4)

    if (len(contours) == 0):
      return
    contour = contours[0]
    left_most = tuple(contour[contour[:, :, 0].argmin()][0])
    right_most = tuple(contour[contour[:, :, 0].argmax()][0])
    top_most = tuple(contour[contour[:, :, 1].argmin()][0])
    bottom_most = tuple(contour[contour[:, :, 1].argmax()][0])

    font = cv2.FONT_HERSHEY_SIMPLEX
    #cv2.putText(img_mask_rgb, 'links', left_most, font, 0.5, (50, 200, 50), 1)
    #cv2.putText(img_mask_rgb, 'rechts', right_most, font, 0.5, (50, 200, 50), 1)
    #cv2.putText(img_mask_rgb, 'oben', top_most, font, 0.5, (50, 200, 50), 1)
    #cv2.putText(img_mask_rgb, 'unten', bottom_most, font, 0.5, (50, 200, 50), 1)


    cv2.rectangle(img_mask_rgb, (bx, by), (bx+bw, by+bh), (50, 50, 200), 2)
    cv2.putText(img_mask_rgb, 'x = ' + str(bx),
      (bx+25, by-60), font, 0.75, (50, 50, 200), 2
    )
    cv2.putText(img_mask_rgb, "y = " + str(by),
      (bx+25, by-25), font, 0.75, (50, 50, 200), 2
    )

    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * peri, True)
    if (len(approx) != 5):
      return

    for point in approx:
      x, y = point[0]
      cv2.line(img_mask_rgb, (x-m, y-m), (x+m, y+m), (50, 150, 50), 2)
      cv2.line(img_mask_rgb, (x+m, y-m), (x-m, y+m), (50, 150, 50), 2)


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
    cv2.putText(img_mask_rgb, 'bottom right',
      (cornerX, cornerY), font, 0.75, (0, 255, 0), 2
    )

    top_left_point = approx[0][0]
    furtherstSq = 0
    for p in approx:
      p = p[0]
      if p[0] == closest[0][0] or p[1] == closest[0][1]:
        continue
      if p[0] == closest[1][0] or p[1] == closest[1][1]:
        continue
      print(cornerX, cornerY, p[1])
      distanceSq = (p[0]-cornerX)*(p[0]-cornerX)+(p[1]-cornerY)*(p[1]-cornerY)
      if distanceSq > furtherstSq:
        furtherstSq = distanceSq
        top_left_point = p

    cv2.putText(img_mask_rgb, 'top left',
      (top_left_point[0], top_left_point[1]), font, 0.75, (0, 255, 0), 2
    )

    degrees = math.degrees(math.atan2(cornerX-top_left_point[0], cornerY-top_left_point[1])) - 45
    cv2.putText(img_mask_rgb, str(int(degrees)) + ' grad',
      (50, 50), font, 0.75, (0, 255, 0), 2
    )


    #img_corners = find_corners(img_fill)


    #img_edged = cv2.Canny(img_fill, 30, 200)
    #_ ,contours, _ = cv2.findContours(img_edged_inverse, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    #img_contours = np.zeros(img_edged.shape)
    #cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    #_, contours, _ = cv2.findContours(img_fill, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #img_contours = np.zeros(img_mask.shape)
    #cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 3)

    #for contour in contours:
      #for point in contour: print(point)
      #peri = cv2.arcLength(contour.astype(np.int), True)
      #approx = cv2.approxPolyDP(contour, 0.04 * peri, True)

      #print(approx, len(approx))
      #if len(approx) == 5:
        #print(approx)



    #corners = cv2.goodFeaturesToTrack(img_fill, 5, 0.01, 20)
    #corners = np.int0(corners)

    #for c in corners:
      #x, y = c.ravel()
      #cv2.circle(img_fill, (x, y), 3, 0, -1)


    #kernel = np.ones((2, 2), np.uint8)
    #img_erosion = cv2.erode(img_mask, kernel, iterations=1)

    #img_gray = cv2.cvtColor(img_hsv, cv2.COLOR_RGB2GRAY)
    #ret, img_thresh = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    #img_contour, contours, hierarchy = cv2.findContours(img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img_contours = np.zeros(img.shape)
    #cv2.drawContours(img_contours, contours, -1, (255, 128, 0), 3)

    #dst = cv2.cornerHarris(img_thresh, 2, 5, 0.04)
    #dst = cv2.dilate(dst, None)

    #img_corners = np.zeros(img.shape)
    #img_corners[dst>0.01*dst.max()]=[0, 0, 255]

    #thresh += 1
    #time.sleep(0.2)
    #print(thresh)

    #img_edges = cv2.Canny(img_mask, 50, 150)
    #lines = cv2.HoughLines(img_edges, 1, np.pi/180, 200)



    #img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #thresh = 50
    #ret, thresh_img = cv2.threshold(img_gray, thresh, 255, cv2.THRESH_BINARY)
    #im2, contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #img_contours = np.zeros(image.shape)
    #cv2.drawContours(img_contours, contours, -1, (0,255,0), 3)

    cv2.imshow('Image', img_mask_rgb) #img_mask_rgb
    cv2.waitKey(2)

    #cv2.destroyWindow('Mask')
    #sys.exit()
  except:
    traceback.print_exc()

rospy.Subscriber('camera/color/image_rect_color', Image, image_callback)
rospy.spin()
