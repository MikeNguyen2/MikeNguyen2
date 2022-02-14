import rospy
from roslib import message
import robot
import cv2 as cv
from sensor_msgs.msg import PointCloud2, Image
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from sensor_msgs import point_cloud2
import tf2_ros

class Commander:
    def __init__(self):
        self.robot = robot.Robot()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.first_callback = True
    
    def __pointcloud_callback(self, point_cloud):
        transform = self.tf_buffer.lookup_transform(
            'cam_d435i', 'base_link', rospy.Time(0)
        )
        transformed_point_cloud = do_transform_cloud(point_cloud, transform)
        points = point_cloud2.read_points(transformed_point_cloud)
        
        n = 0
        avg_x = 0
        avg_y = 0
        avg_z = 0

        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]

            if True:# or z > 0.00 and z < 0.2:
                avg_x += x
                avg_y += y
                avg_z += z
                n += 1
        
        if n < 1:
            return
        
        avg_x /= n
        avg_y /= n
        avg_z /= n

        print('number of points: ', n)
        print('average point: ', avg_x, avg_y, avg_z)

        x = -avg_x - 0.0177
        y = avg_y + 0.61
        if self.first_callback:
            self.robot.move_e(x, y, 0.06, 0, 0, 0)
            self.robot.move_hand(30)
        self.first_callback = False


    def demo(self):
        self.robot.set_speed(velocity=0.4, acceleration=0.4)
        self.robot.move_e(0.045, 0.25, 0.3, 0, 0, 0)

        rospy.Subscriber('/camera/depth_registered/points', PointCloud2, self.__pointcloud_callback)
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('pick_and_place')
    commander = Commander()
    commander.demo()
