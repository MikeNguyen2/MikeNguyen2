import rospy
import bot
from sensor_msgs.msg import PointCloud2, Image
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs import point_cloud2
import tf2_ros

rospy.init_node('move_group_interface_bot')
cobot = bot.MoveGroupInterfaceBot()
cobot.move_h(30, 100)
cobot.move_e(0.045, 0.2, 0.2, 0, 0, 0)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

first_callback = True

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
      #avg_r = 0
      #avg_g = 0
      #avg_b = 0
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
        if z > 0.01 and z < 0.04: #z > -0.07 and z < -0.04: #-0.01, 0.07
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
      #print(avg_r, avg_g, avg_b)
      global first_callback
      if first_callback:
        x = avg_x - 0.0165
        y = -avg_y + 0.5
        cobot.move_q(x, y, 0.06, 0, 1, 0, 0)
        cobot.move_q(x, y, 0.03, 0, 1, 0, 0)
        cobot.move_h(23, 100)
        cobot.move_q(x, y, 0.06, 0, 1, 0, 0)
        cobot.move_q(0.05, 0.3, 0.06, 0, 1, 0, 0)
        cobot.move_q(0.05, 0.3, 0.031, 0, 1, 0, 0)
        cobot.move_h(30, 100)
        cobot.move_q(0.05, 0.3, 0.1, 0, 1, 0, 0)
        first_callback = False


#cobot.move_q(0.07, 0.25, 0.2, 0, 1, 0, 0)
rospy.Subscriber('/camera/depth_registered/points', PointCloud2, pointcloud_callback)
rospy.spin()
