import rospy
import robot
import scene
import time
import sys

import cv2 as cv
import camera
import threading 
import os

class Commander:
    def __init__(self):
        self.robot = robot.Robot()
        self.scene = scene.Scene(
            '/opt/ros/melodic/share/denso_robot_descriptions/cobotta_description/'
        )
        self.cam = camera.Camera()
        print('camera ready')
        self.limit = 1.5 #2.617994
        self.robot.move_j(- self.limit, -0.5396615326699308, 2.0848556011276558, -0.0007266796377727457, 1.5958209820234335, 0.46607877905307055)

    def calibration(self):
        self.robot.set_speed(velocity=1, acceleration=0.4)
        
        self.robot.move_q(-0.0455,-0.1224,0.4467,0.9751,-0.0050,-0.0015,0.2216)
        self.robot.move_q(-0.0455,-0.1224,0.4467,0.9898,-0.0050,-0.0011,0.1422)
        self.robot.move_q(-0.0455,-0.1224,0.4467,0.9975,-0.0051,-0.0007,0.0703)

    def search(self):
        self.robot.set_speed(velocity=0.1, acceleration=1)

        start = time.time()
        self.robot.move_j(self.limit, -0.5396615326699308, 2.0848556011276558, -0.0007266796377727457, 1.5958209820234335, 0.46607877905307055)
        self.robot.move_j(self.limit, 0.12137211247492546, 1.5512946352874215, -0.0008304910145974237, 1.4682358971982798, 0.22707068671292863)
        self.robot.move_j(-self.limit, 0.12137211247492546, 1.5512946352874215, -0.0008304910145974237, 1.4682358971982798, 0.22707068671292863)
        self.robot.move_j(-self.limit, 0.9248952911941696, 0.37255523095973614, 0.00035295868120390503, 1.843906702236175, 0.15128657621566458)
        self.robot.move_j(self.limit, 0.9248952911941696, 0.37255523095973614, 0.00035295868120390503, 1.843906702236175, 0.15128657621566458)
        
        
        print(time.time() - start)

    def get_training_images(self):
        i = 0
        batch = str(30)
        try:
            os.mkdir('/home/mike/training/batch_' + batch)
        except:
            print('batch already in use')
            return

        for j in range(70):
            time.sleep(1)
            training_image = self.cam.get_image()
            path2 = '/home/mike/training/batch_' + batch + '/training_image' + str(i) +'.jpeg'
            cv.imwrite(path2, training_image)            
            i+=1
            image = cv.imread('/home/mike/training/batch_' + batch + '/training_image' + str(i) +'.jpeg')
            
            if image is not None:
                cv.imshow('image',image)
                
            if cv.waitKey(1) & 0xFF == ord('q'):
                sys.exit()
        sys.exit()
        

if __name__ == '__main__':
    rospy.init_node('calibration_commander_main')
    commander = Commander()
    commander.search()
    # print('go')
    # time.sleep(4)
    # threads = []
    # t = threading.Thread(target=commander.search)
    # s = threading.Thread(target=commander.get_training_images)
    # threads.append(t)
    # threads.append(s)
    # t.start()
    # s.start()
    #commander.search()
    #commander.get_training_images()
    '''lines = set()
    while True:
        _ = raw_input('press to continue')
        pose = commander.robot.group.get_current_pose().pose
        p = pose.position
        o = pose.orientation
        line = 'self.robot.move_q({:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f})'.format(p.x, p.y, p.z, o.x, o.y, o.z, o.w)
        lines.add(line)
        print(lines)'''