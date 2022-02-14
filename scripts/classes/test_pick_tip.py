"""This document combines several classes to create a fluent sequence."""
import rospy
import robot
import camera
import pipette
import new_classify_box_yellow as classifier
import painter
import scene
import cv2 as cv


class Commander:
    """
    Create the commander-class with an inbuilt sequences.

    By creating this object many objects will be genereated.
    """

    def __init__(self):
        """Initialize the commander by creating its tools."""
        self.robot = robot.Robot()
        self.camera = camera.Camera()
        self.pipette = pipette.GilsonPipette()
        self.classifier = classifier.Classifier()
        self.painter = painter.Painter()
        self.scene = scene.Scene(
            '/opt/ros/melodic/share/denso_robot_descriptions/cobotta_description/'
        )
        self.robot.set_speed(0.2,1)

    def pick_tip(self):
        """Execute the sequence below."""
        self.robot.move_e(0.045, 0.25, 0.3, 0, 0, 0)
        box_yellow = None
        while box_yellow is None:
            (self.color_image, self.depth_frame) = self.camera.get_images()
            image_drawn = self.color_image.copy()
            box_yellow = self.classifier.classify_box_yellow(self.color_image, self.depth_frame)

            cv.imshow('Original', self.color_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        (box_points, contour, pattern) = box_yellow
        image_drawn = self.classifier.draw_rec_with_circ(image_drawn, contour, pattern, 'Box_Yellow')  
        print('press a button!')
        cv.imshow('drawing', image_drawn)   
        cv.waitKey(0)

        (x,y) = (0, 719) #pattern[0][0][:2]
        t_x, t_y, t_z = self.camera.get_position()
        distance = self.depth_frame.get_distance(x, y)
        print('distance: ', distance)
        print('cam:  ',  t_x, t_y, t_z )
        real_x, real_y,real_z = self.camera.calculate_real_xyz((t_x, t_y, t_z), x, y, 0.4)
        print('real: ',real_x, real_y,real_z)

        self.robot.move_e(real_x - 0.02, real_y + 0.01, real_z + 0.15, -90, -90, 0) # x + 0.004
    
    def test_robot(self):
        self.robot.move_e(0.1 , 0.2, 0.2, -90, -90, 0)

if __name__ == '__main__':
    rospy.init_node('commander_main')
    commander = Commander()
    commander.pick_tip()
    #commander.test_robot()
    
    
