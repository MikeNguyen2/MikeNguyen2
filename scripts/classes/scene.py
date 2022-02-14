"""This document describes the world, in which the robot performs actions."""

import rospy
import moveit_commander
import geometry_msgs
import time
import math
import tf


class Scene:
    """Create an Environment for the robot."""

    def __init__(self, mesh_directory):
        """
        Initialize the Environment.

        mesh_directory: String,     a path to the mesh directory
        """
        self.mesh_directory = mesh_directory
        self.scene = moveit_commander.PlanningSceneInterface()

    def add_mesh(self, name, x, y, z, rx, ry, rz, model_name):
        """
        Add a 3d object to the robot/scene in RViz.

        name:       String, what the object is going to be called
        x, y, z:    int,    offsets along the respective axis
        model_name: String, what the 3d model of the object is called
        """
        pose_stamped = geometry_msgs.msg.PoseStamped()
        pose_stamped.header.frame_id = 'world'
        pose_stamped.pose.position.x = x
        pose_stamped.pose.position.y = y
        pose_stamped.pose.position.z = z

        quaternion = tf.transformations.quaternion_from_euler(
            math.radians(rx),
            math.radians(ry),
            math.radians(rz)
        )

        pose_stamped.pose.orientation.x = quaternion[0]
        pose_stamped.pose.orientation.y = quaternion[1]
        pose_stamped.pose.orientation.z = quaternion[2]
        pose_stamped.pose.orientation.w = quaternion[3]

        path = self.mesh_directory + model_name
        self.scene.add_mesh(name, pose_stamped, path)
        time.sleep(0.8)

    def attach_mesh(self, parent, name):
        """
        Sticks a mesh object to another mesh object.

        For example: Can be used if the robot grabbed something.
        name:       String, frame_id of the object being "glued"
        parent:     String, frame_id to which the object will be "glued" to
        """
        self.scene.attach_mesh(parent, name)
        time.sleep(0.8)

    def remove_mesh(self, name):
        try:
            self.scene.remove_world_object(name)
            print('removed')
        except:
            print('object does not exist!')

if __name__ == '__main__':
    rospy.init_node('scene_main')
    scene = Scene(
        '/opt/ros/melodic/share/denso_robot_descriptions/cobotta_description/'
    )
