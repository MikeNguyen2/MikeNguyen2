from publisher import *
from subscriber import *
import threading
import sys
sys.path.insert(1, '/home/mike/cobot/scripts/classes')
from sensor_msgs.msg import Image
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError
import numpy as np

def test():
    rospy.init_node('test', anonymous=True)
    sub = Subscriber()
    pub = Publisher()

    threads = []
    for i in range(5):
        t = threading.Thread(target=pub.talk,)
        s = threading.Thread(target=sub.listen)
        threads.append(t)
        threads.append(s)
        t.start()
        s.start()

# pub.talk()
# sub.listen()

# human detection
# camera2    -> image2      list
# image2     -> classifier
# classifier -> human?      boolean
# human?     -> commander

# sequence
# commander       -> objects
# camera          -> image
# objects + image -> classifier  -> objects_found? + positions
# objects_found? + positions -> commander -> command + stop?
# for every command 
# command + stop? -> robot -> feedback
# feedback        -> commander 

def callback(data):
    image = bridge.imgmsg_to_cv2(data)#, desired_encoding='passthrough')
    cv.imshow('image', image)
    cv.waitKey(1)

if __name__ == "__main__":
    rospy.init_node('image_msg')
    cam = camera.Camera()
    bridge = CvBridge()
    rate = rospy.Rate(6)

    sub = rospy.Subscriber("chatter", Image, callback)
    pub = rospy.Publisher("chatter", Image, queue_size=10)

    while not rospy.is_shutdown():
        image = cam.get_image()
        #rospy.loginfo(image)
        pub.publish(bridge.cv2_to_imgmsg(image))
        rate.sleep()