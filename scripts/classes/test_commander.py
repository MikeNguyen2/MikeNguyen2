from matplotlib.pyplot import box
import camera
import rospy
import robot
import new_classifier8 as classifier
import cv2 as cv
import time
import threading
import numpy as np
import scene
import new_painter as painter
#from tablet_classifier import *

def auto_limit(image, sigma):
    """
    returns the best limits for the canny algorithm. 
    """
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

def move_to_top_left():
    """
    used to test the roboter/camera calibration to get the offset
    """
    while(True):
        (image, depth_frame) = my_camera.get_images()
        for j in range(5):
            cv.circle(image, (640, int(720.0/4)*j), 8, (255,0,255), 2)
        for i in range(7):
            cv.circle(image, (int(1280.0/6)*i, 360), 8, (255,0,255), 2)
        cv.imshow('image', image)
        
        if cv.waitKey(1) & 0xFF == ord('q'):
            point_x = 213*3
            point_y = 180*2
            depth = depth_frame.get_distance(point_x,point_y)
            print(depth)
            camera_position = my_camera.get_position()
            x,y,z = my_camera.calculate_real_xyz(camera_position, point_x, point_y, depth)

            (a,b,c) = camera_position
            #my_robot.move_e(x-0.02, y, 0.05, 0, 0, 0)
            my_robot.move_e(x, y, 0.05, -90, -90, 0)
            break

def move_rectangle():
    """
    moves in a rectangle.
    """
    my_robot.set_speed(0.05, 0.05, 0.05)
    my_robot.move_e(-0.05, 0.25, 0.3, 0, 0, 0)
    my_robot.move_e(0.05, 0.25, 0.3, 0, 0, 0)

def make_video():
    """
    tests a classifier.
    """
    (color_image, depth_frame) = my_camera.get_images()
    #path = "/home/mike/cobot/scripts/classes/MTP/test12.mp4"
    #width, height = 1280, 720
    #fourcc = cv.VideoWriter_fourcc(*'mp4v')
    #out = cv.VideoWriter(path, fourcc, 10.0, (width, height))

    while(True):
        (color_image, depth_frame) = my_camera.get_images()
        _, _, cam_z = my_camera.get_position()
        cam_z += 0.1
        my_classifier.set_images(color_image, depth_frame)
        contours = my_classifier.get_contours(color_image)
        #color_image = cv.imread('/home/mike/cobot/scripts/classes/MTP/lab4.png')
        image_drawn = color_image.copy()

        boxes_orange = my_classifier.classify_box_orange(cam_z)
        mtps = my_classifier.classify_mtp(cam_z)
        tablet = my_classifier.classify_tablet(contours, cam_z)

        image_drawn = my_painter.draw_mtp(mtps, image_drawn)
        image_drawn = my_painter.draw_box(boxes_orange, image_drawn)
        image_drawn,_,_ = my_classifier.compare_tablet_pattern(contours, tablet, image_drawn, cam_z)

        #out.write(image_drawn)
        
        cv.imshow('frame', color_image)
        cv.imshow('image_drawn', image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    #out.release()
    cv.destroyAllWindows()

def search():
    """
    orientation motion to check the table.
    """
    my_robot.set_speed(velocity=0.07, acceleration=1)
    limit = 0.6
    my_robot.move_j(limit, -0.5396615326699308, 2.0848556011276558, -0.0007266796377727457, 1.5958209820234335, 0.46607877905307055)
    my_robot.move_j(limit, 0.12137211247492546, 1.5512946352874215, -0.0008304910145974237, 1.4682358971982798, 0.22707068671292863)
    my_robot.move_j(-limit, 0.12137211247492546, 1.5512946352874215, -0.0008304910145974237, 1.4682358971982798, 0.22707068671292863)
    my_robot.move_j(-limit, 0.9248952911941696, 0.37255523095973614, 0.00035295868120390503, 1.843906702236175, 0.15128657621566458)
    my_robot.move_j(limit, 0.9248952911941696, 0.37255523095973614, 0.00035295868120390503, 1.843906702236175, 0.15128657621566458)

def move_up_down():
    """
    moves up and down to test the height recognition.
    """
    my_robot.set_speed(velocity=0.05, acceleration=1)
    my_robot.move_e(0.045, 0.25, 0.35, 0, 0, 0)
    my_robot.move_e(0.045, 0.25, 0.1, 0, 0, 0)
    #my_robot.move_e(0.045, 0.25, 0.3, 0, 0, 0)
    my_robot.move_e(0.045, 0.25, 0.2, 0, 0, 0)

def callback(data):
    image = bridge.imgmsg_to_cv2(data)#, desired_encoding='passthrough')
    cv.imshow('image', image)
    cv.waitKey(1)

def pub_sub():
    """
    generates a subscriber and publisher and tests the data transmission.
    """
    from sensor_msgs.msg import Image
    from cv_bridge import CvBridge, CvBridgeError

    bridge = CvBridge()
    rate = rospy.Rate(6)

    sub = rospy.Subscriber("color_image", Image, callback)
    pub = rospy.Publisher("color_image", Image, queue_size=10)

    while not rospy.is_shutdown():
        (color_image, depth_frame) = my_camera.get_images()
        #rospy.loginfo(image)
        pub.publish(bridge.cv2_to_imgmsg(color_image),bridge)
        rate.sleep()

def tablet():
    """
    tests the tablet detection.
    """
    while True:
        x,y,z = my_camera.get_position()
        z+= 0.1
        (color_image, depth_frame) = my_camera.get_images()
        #color_image = cv.imread('/home/mike/cobot/scripts/classes/MTP/approx.jpg')
        image_drawn = color_image.copy()
        contours = my_camera.get_contours(color_image)  
        (tablet,circles) = my_camera.classify_tablet(contours, z)
        if tablet is not None:
            print(tablet)
            image_drawn, _, _ = my_camera.compare_pattern(contours,circles,tablet,image_drawn,z)
        cv.imshow('image_drawn', image_drawn)

        if cv.waitKey(1) & 0xFF == ord('w'):
            avg_list = []
            
            for counter in range(5):
                x,y,z = my_camera.get_position()
                z+= 0.1
                (color_image, depth_frame) = my_camera.get_images()
                #color_image = cv.imread('/home/mike/cobot/scripts/classes/MTP/approx.jpg')
                image_drawn = color_image.copy()
                contours = my_camera.get_contours(color_image)  
                (tablet,circles) = my_camera.classify_tablet(contours, z)
                if tablet is not None:
                    print(tablet)
                    image_drawn, _, _ = my_camera.compare_pattern(contours,circles,tablet,image_drawn,z)
                    (next_a,next_b) = circles[0]
                    depth = depth_frame.get_distance(640,360)
                    if depth == 0:
                        depth = z-0.01
                    camera_position = my_camera.get_position()
                    x,y,z = my_camera.calculate_real_xyz(camera_position, next_a, next_b, depth)
                    print(depth, z-0.01)
                    avg_list.append((x,y,z))
                cv.imshow('image_drawn', image_drawn)

            if len(avg_list) > 0:
                avg_x = 0
                avg_y = 0
                for point in avg_list:
                    x,y,z = point
                    avg_x += x
                    avg_y += y
                avg_x /= len(avg_list)
                avg_y /= len(avg_list)
                my_robot.move_e(avg_x-0.02, avg_y+0.005, 0.010, 0, 0, 0)
                print(avg_x,avg_y)
            else:
                print("no tablet!")
            break

def test_run():
    """
    an example by combining mtp recoginition with well detection.
    """
    my_camera.reset_color_options()
    while(True):
        (color_image, depth_frame) = my_camera.get_images()
        image_drawn = color_image.copy()
        cam_x, cam_y, cam_z = my_camera.get_position()
        cam_z += 0.1

        my_classifier.set_images(color_image, depth_frame)
        contours = my_classifier.get_contours(color_image)
        mtps = my_classifier.classify_mtp(cam_z)
        boxes_orange = my_classifier.classify_box_orange(cam_z)
        tablet = my_classifier.classify_tablet(contours, cam_z)

        image_drawn = my_painter.draw_mtp(mtps, image_drawn)
        image_drawn = my_painter.draw_box(boxes_orange, image_drawn)
        image_drawn, _, _ = my_classifier.compare_tablet_pattern(contours,tablet,image_drawn,cam_z)
        #contours_tablet = my_classifier.get_tablet_contours(color_image)

        cv.imshow('image_drawn', image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            avg_list = []
            for counter in range(10):
                (color_image, depth_frame) = my_camera.get_images()
                image_drawn = color_image.copy()
                cam_x, cam_y, cam_z = my_camera.get_position()
                cam_z += 0.1
                my_classifier.set_images(color_image, depth_frame)
                contours = my_classifier.get_contours(color_image)

                mtps = my_classifier.classify_mtp(cam_z)
                tablet = my_classifier.classify_tablet(contours, cam_z)
                
                image_drawn = my_painter.draw_mtp(mtps, image_drawn)
                image_drawn, num_letter, num_number = my_classifier.compare_tablet_pattern(contours,tablet,image_drawn,cam_z)

                if len(mtps) > 0 and tablet[0] is not None:
                    if num_letter is None:
                        break
                    well = mtps[0][2][num_number-1][num_letter-1]
                    #well = mtps[0][2][0][0]
                    depth = depth_frame.get_distance(well[0], well[1])
                    if depth == 0:
                        depth = z-0.085
                        print("NO DEPTH ESTIMATION: ", depth)
                    x,y,z = my_camera.calculate_real_xyz((cam_x, cam_y, cam_z), well[0], well[1], depth)
                    avg_list.append((x,y,z))
            
            if len(avg_list) > 0:
                avg_x = 0
                avg_y = 0
                div = 0
                for point in avg_list:
                    x,y,z = point
                    avg_x += x
                    avg_y += y
                    div += 1
                avg_x /= div
                avg_y /= div
                my_robot.move_e(avg_x-0.016, avg_y-0.075, 0.25, 0, 0, 0)
                my_robot.move_e(avg_x-0.016, avg_y-0.075, 0.25, -90, -90, 0)
                my_robot.set_speed(velocity=0.25, acceleration=1)
                my_robot.move_e(avg_x-0.016, avg_y-0.075, 0.07+0.055, -90, -90, 0)
                my_robot.set_speed(velocity=0.1, acceleration=1)
                #my_robot.move_e(avg_x-0.016, avg_y-0.075, 0.04+0.055, -90, -90, 0)
                print(avg_x,avg_y)
            else:
                print("no Well!")
            
            break

def pick_tip():
    while True:
        print('objects')
        print('-----------------------------------')
        (ORIGINAL, DEPTH_FRAME) = my_camera.get_images()
        image_drawn = ORIGINAL.copy()
        x,y,z = my_camera.get_position()
        #z=0.2
        z+=0.1
        my_classifier.set_images(ORIGINAL, DEPTH_FRAME)
        contours = my_classifier.get_contours(ORIGINAL)
        boxes_orange = my_classifier.classify_box_orange(z)
        
        ofs_left = 7
        image_drawn = my_painter.draw_box(boxes_orange, image_drawn)

        cv.imshow('ORIGINAL', ORIGINAL)
        cv.imshow('image_drawn', image_drawn)
        

        if cv.waitKey(1) & 0xFF == ord('q'):
            avg_list = []
            
            for counter in range(10):
                x,y,z = my_camera.get_position()
                z+= 0.1
                (color_image, depth_frame) = my_camera.get_images()
                #color_image = cv.imread('/home/mike/cobot/scripts/classes/MTP/approx.jpg')
                image_drawn = color_image.copy()
                my_classifier.set_images(ORIGINAL, DEPTH_FRAME)
                contours = my_classifier.get_contours(ORIGINAL)
                boxes_orange = my_classifier.classify_box_orange(z)
                #print(boxes_blue)
                image_drawn = my_painter.draw_box(boxes_orange, image_drawn)
                if len(boxes_orange) > 0:
                    box = boxes_orange[0]
                    circles = box[2]
                    circle = circles[0][0]
                    (a_x, a_y) = circle[:2]
                    depth = depth_frame.get_distance(a_x,a_y)
                    if depth == 0:
                        depth = z-0.05
                        print("NO DEPTH ESTIMATION: ", depth)
                    else:
                        print("DEPTH: ", depth)
                    camera_position = my_camera.get_position()
                    x,y,z = my_camera.calculate_real_xyz(camera_position, a_x, a_y, depth)
                    avg_list.append((x,y,z))
                cv.imshow('image_drawn', image_drawn)
                cv.waitKey(1)

            if len(avg_list) > 0:
                avg_x = 0
                avg_y = 0
                div = 0
                for point in avg_list:
                    x,y,z = point
                    avg_x += x
                    avg_y += y
                    div += 1
                avg_x /= div
                avg_y /= div
                pos_x = avg_x-0.016
                pos_y = avg_y-0.0775
                my_robot.move_e(pos_x, pos_y, 0.25, 0, 0, 0)
                my_robot.move_e(pos_x, pos_y, 0.25, -90, -90, 0)
                my_robot.set_speed(velocity=0.25, acceleration=1)
                my_robot.move_e(pos_x, pos_y, 0.12+0.025, -90, -90, 0)
                my_robot.set_speed(velocity=0.1, acceleration=1)
                my_robot.move_e(pos_x, pos_y, 0.09+0.025, -90, -90, 0)
                my_robot.move_e(pos_x, pos_y, 0.077+0.025, -90, -90, 0)
                #print(avg_x,avg_y)
                while True:
                    if cv.waitKey(1) & 0xFF == ord('q'):
                        my_robot.move_e(pos_x, pos_y, 0.09+0.025, -90, -90, 0)
                        my_robot.set_speed(velocity=0.25, acceleration=1)
                        my_robot.move_e(pos_x, pos_y, 0.12+0.025, -90, -90, 0)
                        my_robot.set_speed(velocity=0.5, acceleration=1)
                        my_robot.move_e(pos_x, pos_y, 0.25, -90, -90, 0)
                        my_robot.move_e(pos_x, pos_y, 0.25, 0, 0, 0)
                        my_robot.move_e(0.045, 0.175, 0.25, 0, 0, 0)
                        break
            else:
                print("no Tip!")
            break

if __name__ == "__main__":
    rospy.init_node('mtp_look_up')
    my_camera = camera.Camera()
    my_robot = robot.Robot()
    my_classifier = classifier.Classifier()
    my_scene = scene.Scene('/opt/ros/melodic/share/denso_robot_descriptions/cobotta_description/')
    my_painter = painter.Painter()
    time.sleep(1)

    my_robot.set_speed(velocity=0.5, acceleration=1)
    
    my_robot.move_e(0.045, 0.175, 0.25, 0, 0, 0)
    my_robot.move_hand(19.1)

    #pick_tip()
    #test_run()
    #make_video()

    # IMPORTANT manually start roboflow in chrome
    #search()
    #my_robot.move_e(0.045, 0.25, 0.2, 0, 0, 0)
    #cv.waitKey(0)
    #_ = raw_input()
    #my_camera = camera.Camera()
    #make_video()

    # tablet()

    # threads = []
    # t = threading.Thread(target=search)
    # s = threading.Thread(target=make_video)
    # threads.append(t)
    # threads.append(s)
    # t.start()
    # s.start()

    