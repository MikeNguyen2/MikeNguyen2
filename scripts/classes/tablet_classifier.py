"""
Is able to read the needed Well from the tablet-screen
"""
import camera
import numpy as np
import cv2 as cv

#       breite / hoehe
#gesamt 23.5 / 16.5
#top    9.2  / 2
#left   7.5  / 3.7
#right  21   / 3.7
#bot    9.2  / 11.7
#spalten 10/11 = 0.909
#reihen  6.5/7 = 0.909

def convert_height_to_pixels(z_depth, height_real, length_real, width_real):
    """
    Converts the real lengths to length in pixel in relation to the distance between object and camera.
    """
    # height in cm
    img_width = 1280
    img_length = 720
    # relation = 0.44 # 57.5/1280 oder 32/720 (auf Hoehe 41)
    # 164 * 294mm    210mm     240 * 420p   ---> 156 * 280mm
    # 320 * 575mm    410mm     190 * 290p
        
    relation_length = 0.156/0.21#32.0/41
    relation_width  = 0.280/0.21#57.5/41
    cam_to_obj = z_depth - height_real

    new_length = cam_to_obj * relation_length
    new_width  = cam_to_obj * relation_width

    relation1 = new_length / img_length   
    relation2 = new_width / img_width 

    length_pixel = length_real / relation1
    width_pixel = width_real / relation2

    return (length_pixel, width_pixel)

def tablet_pattern(points):
    """
    returns a list of 41 points. 
    0: next Button
    1-12: top row
    13-20: right column
    20-32: bottom row
    33-40: left column
    """
    x1 = np.power(points[0][0]-points[1][0], 2)
    y1 = np.power(points[0][1]-points[1][1], 2)
    x2 = np.power(points[1][0]-points[2][0], 2)
    y2 = np.power(points[1][1]-points[2][1], 2)
    length_a = np.sqrt(x1 + y1)
    length_b = np.sqrt(x2 + y2)

    if length_a > length_b:
        top_left = points[2]
        top_right = points[3]
        bottom_right = points[0]
        bottom_left = points[1]
    else:
        top_left = points[1]
        top_right = points[2]
        bottom_right = points[3]
        bottom_left = points[0]

    width1 = top_right[0]-top_left[0]
    height1 = top_right[1]-top_left[1]
    width2 = bottom_left[0]-top_left[0]
    height2 = bottom_left[1]-top_left[1]
    #print(width1,height1,width2,height2)

    next_x = top_left[0] + 5.2/23.5 * width1 + 14.9/16.5 * width2
    next_y = top_left[1] + 5.2/23.5 * height1 + 15/16.5 * height2
    
    next_x, next_y = int(next_x), int(next_y)

    circles = []
    circles.append((next_x, next_y))
    for i in range(12):
        x_new = top_left[0] + 9.2/23.5 * width1 + 0.909/23.5 * width1 * i + 2.2/16.5 * width2
        x_new = int(x_new)
        y_new = top_left[1] + 9.2/23.5 * height1 + 0.909/23.5 * height1 * i + 2.2/16.5 * height2
        y_new = int(y_new)
        circles.append((x_new,y_new))

    for i in range(8):
        x_new = top_left[0] + 3.8/16.5 * width2 + 0.909/16.5 * width2 * i + 21/23.5 * width1 
        x_new = int(x_new)
        y_new = top_left[1] + 3.8/16.5 * height2 + 0.909/16.5 * height2 * i + 21/23.5 * height1 
        y_new = int(y_new)
        circles.append((x_new,y_new))

    for i in range(12):
        x_new = top_left[0] + 9.2/23.5 * width1 + 0.909/23.5 * width1 * i + 11.7/16.5 * width2 
        x_new = int(x_new)
        y_new = top_left[1] + 9.2/23.5 * height1 + 0.909/23.5 * height1 * i + 11.7/16.5 * height2 
        y_new = int(y_new)
        circles.append((x_new,y_new))

    for i in range(8):
        x_new = top_left[0] + 3.8/16.5 * width2 + 0.909/16.5 * width2 * i + 7.5/23.5 * width1
        x_new = int(x_new)
        y_new = top_left[1] + 3.8/16.5 * height2 + 0.909/16.5 * height2 * i + 7.5/23.5 * height1
        y_new = int(y_new)
        circles.append((x_new,y_new))
    
    return circles #[(center_x,center_y),(top_x,top_y), (bot_x,bot_y), (left_x,left_y), (right_x,right_y)]

def get_contours(color_image):
    """
    returns the contours and shows important images.
    may need readjustment for runtime
    """
    gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
    median = np.median(gblur)

    lower_limit = int(max(0, (1 - 0.33) * median))
    upper_limit = int(min(255, (1 + 0.33) * median))
    canny_old = cv.Canny(gblur, lower_limit, upper_limit)
    kernel_old = np.ones((5,5))
    dilated_old = cv.dilate(canny_old, kernel_old)

    median = np.median(gblur)
    lower_limit = int(max(0, (1 - 0.33) * median))
    upper_limit = int(min(255, (1 + 0.33) * median))
    canny_gblur = cv.Canny(gblur, lower_limit, upper_limit)#
    cv.imshow('canny_gblur', canny_gblur)
    cv.imshow('ORIGINAL', color_image)
    method = cv.RETR_LIST # cv.RETR_EXTERNAL #cv.RETR_LIST #cv.RETR_TREE
    _, contours, _ = cv.findContours(canny_gblur, method, cv.CHAIN_APPROX_NONE)
    return contours

def search_tablet(contours, z_depth):
    """
    returns the contour of the tablet and the pattern.
    """
    a_pix, b_pix = convert_height_to_pixels(z_depth,0.005,0.235,0.165)
    print('------------------')
    print(a_pix,b_pix)
    tablet_contour = None
    circles = None
    for contour in contours:
        minrect = cv.minAreaRect(contour)
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        bottom, left, top, right = box

        width = np.sqrt(np.power(bottom[0]-left[0], 2)+np.power(bottom[1]-left[1], 2))
        height = np.sqrt(np.power(left[0]-top[0], 2)+np.power(left[1]-top[1], 2))
        if a_pix*1.1 > max(width,height) > a_pix*0.95 and b_pix*1.1 > min(width,height) > b_pix*0.95:
            print(width,height)
            #print(w,h,len(approx))
            circles = tablet_pattern(box)
            tablet_contour = contour
            break 
    return (tablet_contour, circles)

def compare_pattern(contours, circles, tablet, image_drawn, z_depth):
    """
    compares the pattern with the contours and draws on the image, then returns the image.
    letters and numbers.
    bars of the current well.
    next-button.
    """
    a_pix, b_pix = convert_height_to_pixels(z_depth,0.005,0.018,0.004)
    c_pix, c_pix = convert_height_to_pixels(z_depth,0.005,0.003,0.003)
    d_pix, e_pix = convert_height_to_pixels(z_depth,0.005,0.029,0.009)
    if circles is None or contours is None:
        return
    
    num_letter = None
    num_number = None

    for circle in circles:
        cv.circle(image_drawn, (circle[0], circle[1]), 5, (255,0,0), 1)

    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        # if c_pix*c_pix*1.5 > w*h > c_pix*c_pix*0.5:#c_pix*c_pix*1.2 > w*h > c_pix*c_pix*0.8:#100 > w*h > 30:
        #     cen_x = x+0.5*w
        #     cen_y = y+0.5*h
        #     for i in range(len(circles)):
        #         #print(circle_calc, cen_x)
        #         a = np.power(circles[i][0]-cen_x, 2)
        #         b = np.power(circles[i][1]-cen_y, 2)
                
        #         distance = np.sqrt(a + b)
        #         if distance < 15:
        #             cv.circle(image_drawn, (int(cen_x), int(cen_y)), 5, (255,255,255), 3)
                    
        #             #print(w,h,c_pix,b_pix)
        #             #print(w*h,distance)

        if a_pix*b_pix*1.4 > w*h > a_pix*b_pix*0.5:#1200 > w*h > 800:
            cen_x = x+0.5*w
            cen_y = y+0.5*h
            for i in range(20):
                #print(circle_calc, cen_x)
                a = np.power(circles[i][0]-cen_x, 2)
                b = np.power(circles[i][1]-cen_y, 2)
                
                distance = np.sqrt(a + b)
                if distance < 20:
                    cv.circle(image_drawn, (int(cen_x), int(cen_y)), 5, (255,0,255), 3)
                    if i <= 12:
                        num_number = i
                    else: 
                        num_letter = i - 12
                    #print(w*h,distance)
        
        elif d_pix*e_pix*1.2 > w*h > d_pix*e_pix*0.5:#1200 > w*h > 800:
            cen_x = x+0.5*w
            cen_y = y+0.5*h
            for i in range(20):
                #print(circle_calc, cen_x)
                a = np.power(circles[i][0]-cen_x, 2)
                b = np.power(circles[i][1]-cen_y, 2)
                
                distance = np.sqrt(a + b)
                if distance < 15:
                    cv.circle(image_drawn, (int(cen_x), int(cen_y)), 5, (0,0,0), 3)

    minrect = cv.minAreaRect(tablet)
    box = cv.boxPoints(minrect)
    box = np.int0(box)

    dic = {
        1: "A",
        2: "B",
        3: "C",
        4: "D",
        5: "E",
        6: "F",
        7: "G",
        8: "H"
    }
    
    cv.drawContours(image_drawn, tablet, -1, (0,0,255), 1)
    cv.drawContours(image_drawn, [box], -1, (255,255,255), 1)

    try:
        let_letter = dic[num_letter]
        text = let_letter + str(num_number)
        cv.putText(image_drawn, text, (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv.LINE_AA)
    except:
        print(num_letter)
        cv.putText(image_drawn, "unreadable", (10,30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv.LINE_AA)
    return image_drawn, num_letter, num_number

def draw():
    pass

if __name__ == "__main__":
    import rospy
    import time
    rospy.init_node('test_tablet') 
    my_camera = camera.Camera()
    time.sleep(1)
    x,y,z = my_camera.get_position()
    #z =0.2
    z+= 0.1
    print(z)
    out = cv.VideoWriter('tablet.mp4',cv.VideoWriter_fourcc(*'mp4v'), 15, (1280,720))
    while True:
        (color_image, depth_frame) = my_camera.get_images()
        #color_image = cv.imread('/home/mike/cobot/scripts/classes/MTP/approx.jpg')
        image_drawn = color_image.copy()

        contours = get_contours(color_image)  
        (tablet,circles) = search_tablet(contours, z)
        if tablet is not None:
            image_drawn,_,_ = compare_pattern(contours,circles,tablet,image_drawn,z)
        
        cv.imshow('result', image_drawn)
        out.write(image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break    
    out.release()