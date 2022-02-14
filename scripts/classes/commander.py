"""This document combines several classes to create a fluent sequence."""
import rospy
import robot
import camera
import pipette
import classifier
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

    def gilson_track_man_demo(self,my_robot):
        """Execute the sequence below."""
        self.robot.move_e(0.045, 0.25, 0.3, 0, 0, 0)

        image = self.camera.get_image()
        microplate = self.classifier.find_microplate(image)

        image_painted = self.painter.paint_microplate(image, microplate)
        cv.imshow('image_painted', image_painted)
        cv.waitKey(0)
        print(microplate)
        print(microplate.wells2d)
        print(microplate.wells2d[0][0])
        x, y, z = microplate.wells2d[0][0]
        my_robot.move_e(x,y,z+10,0,0,0)
    
    
        
if __name__ == '__main__':
    rospy.init_node('commander_main')
    commander = Commander()
    my_robot = robot.Robot()
    commander.gilson_track_man_demo(my_robot)
    
