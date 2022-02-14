from bcap_service.srv import bcap, bcapRequest
from bcap_service.msg import variant
import moveit_commander
import geometry_msgs
import rospy
import math
import copy
import sys
import tf
import time

def all_close(goal, actual, tolerance):
  all_equal = True
  if type(goal) is list:
    for index in range(len(goal)):
      if abs(actual[index] - goal[index]) > tolerance:
        return False

  elif type(goal) is geometry_msgs.msg.PoseStamped:
    return all_close(goal.pose, actual.pose, tolerance)

  elif type(goal) is geometry_msgs.msg.Pose:
    return all_close(
      moveit_commander.conversions.pose_to_list(goal),
      moveit_commander.conversions.pose_to_list(actual),
      tolerance)

  return True


def obtain_bcap_controller_handle():
  rospy.wait_for_service('bcap_service')
  try:
    service = rospy.ServiceProxy('bcap_service', bcap)

    request = bcapRequest()
    request.func_id = 3
    request.vntArgs.append(variant(vt=8, value='b-CAP'))
    request.vntArgs.append(variant(vt=8, value='CaoProv.DENSO.VRC'))
    request.vntArgs.append(variant(vt=8, value='localhost'))
    request.vntArgs.append(variant(vt=8, value=''))

    # Returns HRESULT and vntRet
    response = service(request)

    return response.vntRet.value
  except rospy.ServiceException as e:
    print('Service call failed: ' + str(e))


'''
  Use rospy.init_node('name', anonymouse=True) in respective python file
'''
class MoveGroupInterfaceBot(object):
  def __init__(self, group_name='arm'):
    super(MoveGroupInterfaceBot, self).__init__()

    moveit_commander.roscpp_initialize(sys.argv)

    self.robot = moveit_commander.RobotCommander()
    self.scene = moveit_commander.PlanningSceneInterface()
    self.group = moveit_commander.MoveGroupCommander(group_name)

    file_name = 'bcap_controller_handle.txt'
    handle = obtain_bcap_controller_handle()
    if handle:
      print('obtained new controller handle: ' + str(handle))
      f = open(file_name, 'w')
      f.write(str(handle))
      f.close()

    self.bcap_controller_handle = open(file_name, 'r').read()
    self.set_speed(1, 1)


  def move_j(self, j1, j2, j3, j4, j5, j6):
    joint_goal = self.group.get_current_joint_values()
    joint_goal[0] = j1
    joint_goal[1] = j2
    joint_goal[2] = j3
    joint_goal[3] = j4
    joint_goal[4] = j5
    joint_goal[5] = j6

    self.group.go(joint_goal, wait=True)
    self.group.stop()

    current_joints = self.group.get_current_joint_values()
    return all_close(joint_goal, current_joints, 0.01)


  def move_q(self, x, y, z, qx, qy, qz, qw):
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

    current_pose = self.group.get_current_pose().pose
    return all_close(pose_goal, current_pose, 0.01)


  def move_e(self, x, y, z, rx, ry, rz):
    quat = tf.transformations.quaternion_from_euler(
      math.radians(rx+180), math.radians(ry), math.radians(rz+180))
    return self.move_q(x, y, z, quat[0], quat[1], quat[2], quat[3])


  def move_h(self, gap, speed):
    rospy.wait_for_service('bcap_service')
    try:
      service = rospy.ServiceProxy('bcap_service', bcap)

      request = bcapRequest()
      request.func_id = 17
      request.vntArgs.append(variant(vt=19, value=str(self.bcap_controller_handle)))
      request.vntArgs.append(variant(vt=8, value='HandMoveA'))
      request.vntArgs.append(variant(vt=8195, value=str(gap)+', '+str(speed*100)))

      service(request)
    except rospy.ServiceException as e:
      print('Service call failed: ' + str(e))


  def move_l(self, points):
    waypoints = []
    pose = self.group.get_current_pose().pose

    for point in points:
      waypoint = copy.deepcopy(pose)
      
      waypoint.position.x = point[0]
      waypoint.position.y = point[1]
      waypoint.position.z = point[2]

      quat = tf.transformations.quaternion_from_euler(
        math.radians(point[3]+180), math.radians(point[4]), math.radians(point[5]+180))

      waypoint.orientation.x = quat[0]
      waypoint.orientation.y = quat[1]
      waypoint.orientation.z = quat[2]
      waypoint.orientation.w = quat[3]

      waypoints.append(waypoint)
    
    plan, fraction = self.group.compute_cartesian_path(waypoints, 0.01, 0.0)
    self.group.execute(plan, wait=True)


  '''
    Scaling factors between 0 and 1
  '''
  def set_speed(self, velocity, acceleration):
    self.group.set_max_velocity_scaling_factor(velocity)
    self.group.set_max_acceleration_scaling_factor(acceleration)


if __name__ == '__main__':
  rospy.init_node('move_group_interface_bot')
  b = MoveGroupInterfaceBot()

  b.set_speed(1, 1)
  b.move_j(0.05, 0.1, 1.7, 0.15, 0.2, 0.3)
  b.move_e(0, 0.25, 0.25, 0, 0, 0)
  b.move_q(0, 0.2, 0.2, 0, 1, 0, 0)
  b.move_h(10, 1)
  b.move_h(30, 1)

  b.move_l([
    (0.0, 0.2, 0.25, 0, 0, 0),
    (0.0, 0.2, 0.12, 0, 0, 0),
    (0.0, 0.2, 0.25, 0, 0, 0)
  ])

  b.move_l([
    (0.05, 0.2, 0.25, 0, 0, 0),
    (0.05, 0.2, 0.12, 0, 0, 0),
    (0.05, 0.2, 0.25, 0, 0, 0)
  ])
