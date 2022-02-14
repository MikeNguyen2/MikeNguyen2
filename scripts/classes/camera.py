"""This document describes a realsense Camera."""
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import rospy

class Camera:
    """Create a camera to get images."""

    def __init__(self, name='cam_d435i', width=1280, height=720, fps=6):
        """
        Initialize the camera by setting up the pipeline and tf objects.

        name:   String, name of the camera
        width:  int,    number of pixels on x-axis
        length: int,    number of pixels on y-axis
        fps:    int,    number of images per second
        """
        import tf2_ros
        self.name = name

        pipeline = rs.pipeline()
        config = rs.config()

        stream = rs.stream.color
        format = rs.format.bgr8
        stream_depth = rs.stream.depth
        format_depth = rs.format.z16
        config.enable_stream(stream, width, height, format, fps)
        config.enable_stream(stream_depth, width, height, format_depth, fps)

        self.profile = pipeline.start(config)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        stream_profile = color_frame.profile.as_video_stream_profile()
        self.deprojection_intrinsics = stream_profile.intrinsics

        self.sensor = self.profile.get_device().query_sensors()[1]
        #self.sensor.set_option(rs.option.enable_auto_exposure, True)
        #self.sensor.set_option(rs.option.enable_auto_white_balance, True)

        self.sensor_dep = self.profile.get_device().first_depth_sensor()

        self.pipeline = pipeline

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

    def get_image(self):
        """
        Wait for frames of pipeline.

        return:   array,    a bgr color image
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_images(self):
        """
        Wait for frames of pipeline.

        return:   2arrays,    a bgr color image and a depth image
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame() 
        
        while not depth_frame or not color_frame:
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame() 
            print('no frame')

        color_image = np.asanyarray(color_frame.get_data())
    
        return (color_image,depth_frame)

    def get_position(self):
        """
        Return the current camera positon.

        return  tuple    the real world position of the camera
        """
        transform_stamped = self.tf_buffer.lookup_transform(
            'world', self.name, rospy.Time()
        )
        translation = transform_stamped.transform.translation
        return translation.x, translation.y, translation.z

    def calculate_real_xyz(self, camera_position, image_x, image_y, z_depth):
        """
        Convert the image position to the real world position.

        camera_position: tuple,     the position of the camera
        image_x:    int,            the 2d-x-coordinate
        iamge_y:    int,            the 2d-y-coordinate
        z_depth:    int,            the distance from the camera to the object

        return:     tuple,          the pixel-position in the real world
        """
        pixel = image_x, image_y
        deprojected_point = rs.rs2_deproject_pixel_to_point(
            self.deprojection_intrinsics, pixel, z_depth
        )
        # TODO Use tf library here so orientation is also taken into account
        x = deprojected_point[0] + camera_position[0]  # TODO: Fix x is y
        y = -deprojected_point[1] + camera_position[1]  # TODO: Fix y is x
        z = -deprojected_point[2] + camera_position[2]  # TODO: Fix wrong z value
        #print('cam_pos: ', camera_position)
        #print('point:', deprojected_point)
        #print('point_to_world: ', [x,y,z])
        return x, y, z

    def set_color_options(self,enable_auto_exposure=None, enable_auto_white_balance=None, exposure=None, gain=None, brightness=None, contrast=None, gamma=None, hue=None, saturation=None, sharpness=None, white_balance=None):
        if exposure is not None:   self.sensor.set_option(rs.option.exposure, exposure)
        if gain is not None:       self.sensor.set_option(rs.option.gain, gain)
        if brightness is not None: self.sensor.set_option(rs.option.brightness, brightness)
        if contrast is not None:   self.sensor.set_option(rs.option.contrast, contrast)
        if gamma is not None:      self.sensor.set_option(rs.option.gamma, gamma)
        if hue is not None:        self.sensor.set_option(rs.option.hue, hue)
        if saturation is not None: self.sensor.set_option(rs.option.saturation, saturation)
        if sharpness is not None:  self.sensor.set_option(rs.option.sharpness, sharpness)
        if white_balance is not None: self.sensor.set_option(rs.option.white_balance, white_balance)
        if enable_auto_exposure is not None: self.sensor.set_option(rs.option.enable_auto_exposure, enable_auto_exposure)
        if enable_auto_white_balance is not None: self.sensor.set_option(rs.option.enable_auto_white_balance, enable_auto_white_balance)

    def reset_color_options(self):
        self.sensor.set_option(rs.option.exposure, 166)
        self.sensor.set_option(rs.option.gain, 64)
        self.sensor.set_option(rs.option.brightness, 0)
        self.sensor.set_option(rs.option.contrast, 50)
        self.sensor.set_option(rs.option.gamma, 300)
        self.sensor.set_option(rs.option.hue, 0)
        self.sensor.set_option(rs.option.saturation, 64)
        self.sensor.set_option(rs.option.sharpness, 50)
        self.sensor.set_option(rs.option.white_balance, 4600)
        self.sensor.set_option(rs.option.enable_auto_exposure, True)
        self.sensor.set_option(rs.option.enable_auto_white_balance, True)

    def set_depth_options(self, exposure=None, gain=None, laser_power=None, enable_auto_exposure=None):
        if exposure is not None:    self.sensor_dep.set_option(rs.option.exposure, exposure)
        if gain is not None:        self.sensor_dep.set_option(rs.option.gain, gain)
        if laser_power is not None: self.sensor_dep.set_option(rs.option.laser_power, laser_power)
        if enable_auto_exposure is not None: self.sensor.set_option(rs.option.enable_auto_exposure, enable_auto_exposure)
        #self.sensor_dep.set_option(rs.option.confidence_threshold, peak_threshold)
        #self.sensor_dep.set_option(rs.option.disparity, disparity)
        #self.sensor_dep.set_option(rs.option.remove_threshold, remove_threshold)

    def reset_depth_options(self):
        self.sensor_dep.set_option(rs.option.exposure, 8500)
        self.sensor_dep.set_option(rs.option.gain, 16)
        self.sensor_dep.set_option(rs.option.enable_auto_white_balance, True)
        self.sensor_dep.set_option(rs.option.laser_power, 150)

class Camera2:
    """Create a camera to get images."""

    def __init__(self, name='cam_d435i', width=1280, height=720, fps=6):
        """
        Initialize the camera by setting up the pipeline and tf objects.

        name:   String, name of the camera
        width:  int,    number of pixels on x-axis
        length: int,    number of pixels on y-axis
        fps:    int,    number of images per second
        """
        self.name = name

        pipeline = rs.pipeline()
        config = rs.config()

        stream = rs.stream.color
        format = rs.format.bgr8
        stream_depth = rs.stream.depth
        format_depth = rs.format.z16
        config.enable_stream(stream, width, height, format, fps)
        config.enable_stream(stream_depth, width, height, format_depth, fps)

        profile = pipeline.start(config)

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        stream_profile = color_frame.profile.as_video_stream_profile()
        self.deprojection_intrinsics = stream_profile.intrinsics

        sensor = profile.get_device().query_sensors()[1]
        sensor.set_option(rs.option.enable_auto_exposure, True)
        sensor.set_option(rs.option.enable_auto_white_balance, True)

        if color_frame.supports_frame_metadata(rs.frame_metadata_value.actual_exposure):
            exposure = color_frame.get_frame_metadata(rs.frame_metadata_value.actual_exposure)
        #exposure = rs . option . exposure 
            print(exposure)
        else:
            print('can not read exposure')
        #self.sensor.set_option(rs.option.white_balance, a)
        self.pipeline = pipeline

        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)

    def get_image(self):
        """
        Wait for frames of pipeline.

        return:   array,    a bgr color image
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        image = np.asanyarray(color_frame.get_data())
        return image

    def get_images(self):
        """
        Wait for frames of pipeline.

        return:   2arrays,    a bgr color image and a depth image
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = None
        color_frame = None
        
        while not depth_frame or not color_frame:
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame() 

        color_image = np.asanyarray(color_frame.get_data())
        
        # depth_image = []
        # for i in range (720):
        #     depth_image.append([])
        #     for j in range(1280):
        #         depth_image[i].append(depth_frame.get_distance(j,i))
        return (color_image,depth_frame)

if __name__ == '__main__':
    rospy.init_node('camera_node')
    cam = Camera()
    # careful when using set! you may need to reset the camera afterwards.
    #cam.set(1)
    time.sleep(1)
    #print(cam.get_position())
    #cv.imwrite('image.png', cam.get_image())
    #x,y,z = cam.get_position()
    #print(x,y,z)
    #cam.set_color_options(brightness=-64,contrast=0,gamma=100,hue=50,saturation=100,sharpness=50,white_balance=2800)
    #cam.reset_color_options()
    #cam.set_depth_options()
    while True:
        #(color_img,depth_frame) = cam.get_images()
        #print(depth_frame.get_distance(640,360))
        # depth_img = []
        # for i in range (720):
        #     depth_img.append([])
        #     for j in range(1280):
        #         depth_img[i].append(depth_frame.get_distance(j,i))
        # scaledarray = (depth_img/np.max(depth_img))*255
        
        # cv.imshow('depth', scaledarray)

        color_img = cam.get_image()
        cv.imshow('color', color_img)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
