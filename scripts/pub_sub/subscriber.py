import rospy
from std_msgs.msg import String, Bool
import sys
sys.path.insert(1, '/home/mike/cobot/scripts/classes')
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

class Subscriber():
    def __init__(self):
        self.br = CvBridge()

    def callback(self, data):
        # print()
        # print()
        # print(data)
        try:
            image = self.br.imgmsg_to_cv2(data)#, desired_encoding='passthrough')
            rospy.loginfo(image)
            cv.imshow('image',image)
            cv.waitKey(1)
        except CvBridgeError as e:
            print(e)
        # print('clear2')
        print(data)
        

    def listen(self):
        rospy.Subscriber("chatter", Image, self.callback)
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    sub = Subscriber()
    sub.listen()