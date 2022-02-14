import tf.transformations
import traceback
import tf2_ros
import rospy
import math
import bot

rospy.init_node('transform_test')
robot = bot.MoveGroupInterfaceBot()
robot.move_e(0.045, 0.25, 0.3, 0, 0, 0)

tf_buffer = tf2_ros.Buffer()
tf_listener = tf2_ros.TransformListener(tf_buffer)

rate = rospy.Rate(10.0)
while not rospy.is_shutdown():
    try:
        o1 = 'world'
        o2 = 'cam_d435i'
        transform_stamped = tf_buffer.lookup_transform(o1, o2, rospy.Time())
        #print(transform_stamped)
        
        translation = transform_stamped.transform.translation
        x = translation.x
        y = translation.y
        z = translation.z

        rotation = transform_stamped.transform.rotation
        quaternion = [rotation.x, rotation.y, rotation.z, rotation.w]
        roll, pitch, yaw = tf.transformations.euler_from_quaternion(quaternion)
        roll = math.degrees(roll)
        pitch = math.degrees(pitch)
        yaw = math.degrees(yaw)

        print('x', x)
        print('y', y)
        print('z', z)
        print('r', roll)
        print('p', pitch)
        print('y', yaw)
    except:
        traceback.print_exc()
        continue

    rate.sleep()