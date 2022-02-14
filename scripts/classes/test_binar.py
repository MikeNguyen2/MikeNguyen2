import camera
import cv2 as cv
import numpy as np

lower = 0
upper = 255

def on_change_lower(value):
  global lower
  lower = value

def on_change_upper(value):
  global upper
  upper = value

def __auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

def show_auto_canny():
    (color_image, depth_frame) = my_camera.get_images()
    gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

    lower_blur, upper_blur = __auto_limit(gblur, 0.33)
    canny_blur = cv.Canny(gblur, lower_blur, upper_blur)

    lower_canny, upper_canny = __auto_limit(gray, 0.33)
    canny = cv.Canny(gray, lower_canny, upper_canny)
    
    cv.imshow('ORIGINAL', color_image)
    cv.imshow('gblur', gblur)
    cv.imshow('canny_blur', canny_blur)
    cv.imshow('gray', gray)
    cv.imshow('canny', canny)

def show_canny_picker():
    cv.namedWindow('Controls')
    slider_lower = cv.createTrackbar('lower', 'Controls', lower, 255, on_change_lower)
    slider_upper = cv.createTrackbar('upper', 'Controls', upper, 255, on_change_upper)
    (color_image, depth_frame) = my_camera.get_images()
    gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
    canny = cv.Canny(gray, lower, upper)
    cv.imshow('canny2', canny)

        

if __name__ == '__main__':
    my_camera = camera.Camera2()
    while True:
        #show_auto_canny()
        show_canny_picker()
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    