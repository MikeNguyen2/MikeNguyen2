"""This document describes a roboter's actions."""
from bcap_service.srv import bcap, bcapRequest
from bcap_service.msg import variant
import moveit_commander
import geometry_msgs
import rospy
import math
import copy
import sys
import tf


class Robot:
    """
    Represent the robot and provides methods for controlling the arm and hand.

    Important: Use rospy.init_node('name') before using this class
    """

    def __init__(self, group_name='arm', end_effector_name='J6'):
        """
        Generate the robot objects.

        group_name: String, name of the move group defined in urdf file
        end_effector_name: String, label of the last joint of the robot
        """
        self.group_name = group_name
        self.end_effector_name = end_effector_name

        moveit_commander.roscpp_initialize(sys.argv)
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group = moveit_commander.MoveGroupCommander(group_name)

        handle_file_name = 'controller_handle'
        handle = self.__obtain_controller_handle()
        if handle:
            print('obtained new controller handle via bcap: ' + str(handle))
            file = open(handle_file_name, 'w')
            file.write(str(handle))
            file.close()

        self.controller_handle = open(handle_file_name, 'r').read()

        self.velocity = 1
        self.acceleration = 1
        self.hand_velocity = 1

    def __obtain_controller_handle(self):
        """
        Use Denso's bcap interface to get the value of the controller handle.

        It is needed to control the electric gripper of the robot.
        returns: int, controller handle.
        """
        rospy.wait_for_service('bcap_service')
        try:
            service = rospy.ServiceProxy('bcap_service', bcap)

            request = bcapRequest()
            request.func_id = 3
            request.vntArgs.append(variant(vt=8, value='b-CAP'))
            request.vntArgs.append(variant(vt=8, value='CaoProv.DENSO.VRC'))
            request.vntArgs.append(variant(vt=8, value='localhost'))
            request.vntArgs.append(variant(vt=8, value=''))

            response = service(request)  # response includes HRESULT and vntRet

            return response.vntRet.value
        except rospy.ServiceException as e:
            print('Service call failed: ' + str(e))

    def move_j(self, j1, j2, j3, j4, j5, j6):
        """
        Move the robot to a specified joint goal.

        j1, j2, j3, j4, j5, j6 are floats and stand for the rotation per joint.
        """
        joint_goal = self.group.get_current_joint_values()
        joint_goal[0] = j1
        joint_goal[1] = j2
        joint_goal[2] = j3
        joint_goal[3] = j4
        joint_goal[4] = j5
        joint_goal[5] = j6

        self.group.go(joint_goal, wait=True)
        self.group.stop()

    def move_q(self, x, y, z, qx, qy, qz, qw):
        """
        Move the robot to a specified cartesian goal.

        x, y, z are offsets on the three axis.
        qx, qy, qz, qw are components of the orientation quaternion.
        """
        pose_goal = geometry_msgs.msg.Pose()

        pose_goal.position.x = x
        pose_goal.position.y = y
        pose_goal.position.z = z

        pose_goal.orientation.x = qx
        pose_goal.orientation.y = qy
        pose_goal.orientation.z = qz
        pose_goal.orientation.w = qw

        self.group.set_pose_target(pose_goal)
        self.group.go(wait=True)
        self.group.stop()
        self.group.clear_pose_targets()

    def move_e(self, x, y, z, rx, ry, rz):
        """
        Move the robot to a specified cartesian goal.

        x, y, z are offsets on the three axis.
        rx, ry, rz, resemble rotation in euler angles of the end effector
        """
        quat = tf.transformations.quaternion_from_euler(
            math.radians(rx+180), math.radians(ry), math.radians(rz+180))
        self.move_q(x, y, z, quat[0], quat[1], quat[2], quat[3])

    def move_le(self, points):
        """
        Move the robot along points by computing a linear path between them.

        Orientation is given in euler angles.
        points: list of tuples, each representing a point in 3D space
        """
        waypoints = []
        pose = self.group.get_current_pose().pose

        for point in points:
            waypoint = copy.deepcopy(pose)

            waypoint.position.x = point[0]
            waypoint.position.y = point[1]
            waypoint.position.z = point[2]

            quaternion = tf.transformations.quaternion_from_euler(
                math.radians(point[3]+180),
                math.radians(point[4]),
                math.radians(point[5]+180)
            )

            waypoint.orientation.x = quaternion[0]
            waypoint.orientation.y = quaternion[1]
            waypoint.orientation.z = quaternion[2]
            waypoint.orientation.w = quaternion[3]

            waypoints.append(waypoint)

        plan, _ = self.group.compute_cartesian_path(waypoints, 0.001, 0.0)

        plan = self.group.retime_trajectory(
            self.group.get_current_state(), plan,
            self.velocity, self.acceleration,
            'time_optimal_trajectory_generation'
            # default is 'iterative_time_parameterization'
        )

        self.group.execute(plan, wait=True)

    def move_lq(self, points):
        """
        Move the robot along points by computing a linear path between them.

        Orientation is given by a quaternion.
        points: list of tuples, each representing a point in 3D space
        """
        waypoints = []
        pose = self.group.get_current_pose().pose

        for point in points:
            waypoint = copy.deepcopy(pose)

            waypoint.position.x = point[0]
            waypoint.position.y = point[1]
            waypoint.position.z = point[2]

            waypoint.orientation.x = point[3]
            waypoint.orientation.y = point[4]
            waypoint.orientation.z = point[5]
            waypoint.orientation.w = point[6]

            waypoints.append(waypoint)

        plan, _ = self.group.compute_cartesian_path(waypoints, 0.001, 0.0)

        plan = self.group.retime_trajectory(
            self.group.get_current_state(), plan,
            self.velocity, self.acceleration,
            'time_optimal_trajectory_generation'
            # default is 'iterative_time_parameterization'
        )

        self.group.execute(plan, wait=True)

    def move_hand(self, gap):
        """
        Move the two fingers of the robot.

        gap: float in millimeters between 0 and 30 (distance between fingers)
        """
        rospy.wait_for_service('bcap_service')
        try:
            service = rospy.ServiceProxy('bcap_service', bcap)

            request = bcapRequest()
            request.func_id = 17

            value_19 = str(self.controller_handle)
            request.vntArgs.append(variant(vt=19, value=value_19))

            value_8 = 'HandMoveA'
            request.vntArgs.append(variant(vt=8, value=value_8))

            value_8195 = str(gap)+', '+str(self.hand_velocity*100)
            request.vntArgs.append(variant(vt=8195, value=value_8195))

            service(request)
        except rospy.ServiceException as e:
            print('Service call failed: ' + str(e))

    def set_speed(self, velocity=None, acceleration=None, hand_velocity=None):
        """
        Adjust the pace of the robot.

        All parameters are scaling factors between 0 and 1.
        """
        if velocity:
            self.velocity = velocity
            self.group.set_max_velocity_scaling_factor(velocity)
        if acceleration:
            self.acceleration = acceleration
            self.group.set_max_acceleration_scaling_factor(acceleration)
        if hand_velocity:
            self.hand_velocity = hand_velocity


if __name__ == '__main__':
    rospy.init_node('robot_main')
    bot = Robot()
    bot.set_speed(0.1, 0.1, 0.1)
    print(bot.group.get_current_pose().pose)
