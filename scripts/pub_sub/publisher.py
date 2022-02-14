import rospy
from std_msgs.msg import String, Bool
import sys
sys.path.insert(1, '/home/mike/cobot/scripts/classes')
import camera
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge

class Publisher:
    def __init__(self):
        self.cam = camera.Camera()
        self.br = CvBridge()

    def talk(self):
        pub = rospy.Publisher('chatter', Image, queue_size=10)
        rate = rospy.Rate(6) # in hz (messages per second)
        while not rospy.is_shutdown():
            image = self.cam.get_image()
            rospy.loginfo(image)
            pub.publish(self.br.cv2_to_imgmsg(image))
            # a = True
            # rospy.loginfo(a)
            # pub.publish(a)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('talker', anonymous=True)
    pub = Publisher()
    try:
        pub.talk()
    except rospy.ROSInterruptException:
        pass