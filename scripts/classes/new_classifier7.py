import camera
import painter
import cv2 as cv
import numpy as np
import new_painter as painter
import time

class Classifier:

    def __init__(self):
        
        # mtp
        self.mtp_ofs = 10
        self.mtp_w_min = 270 - self.mtp_ofs
        self.mtp_w_max = 290 + self.mtp_ofs
        self.mtp_h_min = 180 - self.mtp_ofs
        self.mtp_h_max = 200 + self.mtp_ofs
        self.mtp_z_min = 0.38
        self.mtp_z_max = 0.4

        self.mtp_dist = 10
        self.mtp_canny_limit = 100
        self.mtp_lower_limit = 10
        self.mtp_min_radius = 8
        self.mtp_max_radius = 12

        # box_blue
        self.box_blue_ofs = 30
        self.box_blue_w_min = 310 - self.box_blue_ofs
        self.box_blue_w_max = 330 + self.box_blue_ofs
        self.box_blue_h_min = 210 - self.box_blue_ofs
        self.box_blue_h_max = 230 + self.box_blue_ofs
        self.box_blue_z_min = 0.32
        self.box_blue_z_max = 0.36

        self.box_blue_dist = 15
        self.box_blue_canny_limit = 75
        self.box_blue_lower_limit = 5
        self.box_blue_min_radius = 10
        self.box_blue_max_radius = 16

        # box_yellow
        self.box_yellow_ofs = 5
        self.box_yellow_w_min = 290 - self.box_yellow_ofs
        self.box_yellow_w_max = 310 + self.box_yellow_ofs
        self.box_yellow_h_min = 190 - self.box_yellow_ofs
        self.box_yellow_h_max = 210 + self.box_yellow_ofs
        self.box_yellow_z_min = 0.35
        self.box_yellow_z_max = 0.37

        self.box_yellow_dist = 10
        self.box_yellow_canny_limit = 50
        self.box_yellow_lower_limit = 10
        self.box_yellow_min_radius = 5
        self.box_yellow_max_radius = 10
    
    def __clahe(self, bgr_image):
        hsv = cv.cvtColor(bgr_image, cv.COLOR_BGR2HSV)
        hsv_planes = cv.split(hsv)
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        hsv_planes[2] = clahe.apply(hsv_planes[2])
        hsv = cv.merge(hsv_planes)
        return cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    def __get_binar(self, color_image, spectrum):
        lower_limit = np.asarray(spectrum[0:3])
        upper_limit = np.asarray(spectrum[3:6])
        return cv.inRange(color_image,lower_limit,upper_limit) 

    def __has_corners(self, contour, conrer_min, corner_max):
        length = cv.arcLength(contour, False)
        approx = cv.approxPolyDP(contour, 0.02*length, False)
        corners = len(approx)
        if conrer_min <= corners <= corner_max:
            return True

    def __convert_height_to_pixels(self, z_depth, height_real, length_real, width_real):
        #height in cm
        img_width = 1280
        img_length = 720
        # relation = 0.44 # 57.5/1280 oder 32/720 (auf Hoehe 41)
        # 164 * 294mm    210mm     240 * 420p
        # 320 * 575mm    410mm     190 * 290p
            
        relation_length = 0.164/0.21#32.0/41
        relation_width  = 0.294/0.21#57.5/41
        cam_to_obj = z_depth - height_real

        new_length = cam_to_obj * relation_length
        new_width  = cam_to_obj * relation_width

        relation1 = new_length / img_length   
        relation2 = new_width / img_width 

        length_pixel = length_real / relation1
        width_pixel = width_real / relation2

        return (length_pixel, width_pixel)

    def __auto_limit(self, image, sigma):
        median = np.median(image)
        lower_limit = int(max(0, (1 - sigma) * median))
        upper_limit = int(min(255, (1 + sigma) * median))
        return lower_limit, upper_limit

    def __simple_pattern(self, points, rows, columns, margin1, margin2, radius):
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

        width1 = float(max(np.absolute(top_right[0]-top_left[0]), 1))
        height1 = float(max(np.absolute(top_right[1]-top_left[1]), 1))
        width2 = float(max(np.absolute(bottom_left[0]-top_left[0]), 1))
        height2 = float(max(np.absolute(bottom_left[1]-top_left[1]), 1))

        top_form = np.power(margin1, 2)*(np.power(width1, 2)+np.power(height1, 2))
        bottom_form = 1 + np.power(height1/width1, 2)
        margin_x = np.absolute(np.sqrt(top_form/bottom_form))
        margin_y = margin_x * height1/width1

        top_form2 = np.power(margin2, 2)*(np.power(width2, 2)+np.power(height2, 2))
        bottom_form2 = 1 + np.power(height2/width2, 2)
        margin_x2 = np.absolute(np.sqrt(top_form2/bottom_form2))
        margin_y2 = margin_x2 * height2/width2

        color = (0, 255, 0)
        circles = []
        if length_a > length_b:
            for column in range(columns):
                circles.append([])
                growth_x = (width1 - 2*margin_x)/(columns-1)*column
                growth_y = (height1 - 2*margin_y)/(columns-1)*column
                start_x = top_left[0] + margin_x + growth_x
                start_y = top_left[1] + margin_y + growth_y

                for row in range(rows):
                    growth2_x = (width2 - 2*margin_x2)/(rows-1)*row
                    grpwth2_y = (height2 - 2*margin_y2)/(rows-1)*row
                    pos_x = start_x - margin_x2 - growth2_x
                    pos_y = start_y + margin_y2 + grpwth2_y
                    circles[column].append([int(pos_x), int(pos_y), radius, color, 0])
        else:
            for column in range(columns):
                circles.append([])
                growth_x = (width1 - 2*margin_x) / (columns - 1) * column
                growth_y = (height1 - 2*margin_y) / (columns - 1) * column
                start_x = top_left[0] + margin_x + growth_x
                start_y = top_left[1] - margin_y - growth_y

                for row in range(rows):
                    growth2_x = (width2 - 2*margin_x2) / (rows - 1) * row
                    growth2_y = (height2 - 2*margin_y2) / (rows - 1) * row
                    pos_x = start_x + margin_x2 + growth2_x
                    pos_y = start_y + margin_y2 + growth2_y
                    circles[column].append([int(pos_x), int(pos_y), radius, color, 0])
        return circles

    def __has_lengths(self, width, height, w_min, w_max, h_min, h_max):
        if w_max < max(width, height) or max(width, height) < w_min:
            return False
        if h_max < min(width, height) or min(width, height) < h_min:
            return False
        return True

        # if w_max >= max(width, height) >= w_min and h_max >= min(width, height) >= h_min:
        #     return True
        # return Flase

    def __is_centered(self, center_x, center_y):
        offset = 1000
        img_cen_x = 640
        img_cen_y = 360
        if img_cen_x-offset > center_x or center_x > img_cen_x+offset:
            return False
        if img_cen_y-offset > center_y or center_y > img_cen_y+offset:
            return False
        return True

        # if img_cen_x + offset >= center_x >= img_cen_x - offset and img_cen_y + offset >= center_y >= img_cen_y - offset:
        #     return True
        # return False

    def __has_circles(self, img, dist, canny_limit, lower_limit, min_r, max_r, x, y):
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
        dilated = cv.dilate(gblur, (5, 5))
        dilated = cv.erode(dilated, (5,5))
        circles_zoom = cv.HoughCircles(dilated, cv.HOUGH_GRADIENT, 1, dist,
                                    param1=canny_limit, param2=lower_limit,
                                    minRadius=min_r, maxRadius=max_r)
        circles_seen = []
        if circles_zoom is not None:
            for circle_zoom in circles_zoom[0]:
                new_x = int(circle_zoom[0]+x)
                new_y = int(circle_zoom[1]+y)
                radius = int(circle_zoom[2])
                circle_new = (new_x, new_y, radius)
                circles_seen.append(circle_new)

        # if len(circles_seen) < 5:
        #     return False
        return circles_seen

    def __compare_pattern(self, pattern, circles_seen, dist_to_circ):
        new_pattern = pattern[:]
        for row in range(len(pattern)):
            for column in range(len(pattern[0])):
                circle_calc = pattern[row][column]
                for circle_seen in circles_seen:
                    a = np.power(circle_calc[0]-circle_seen[0], 2)
                    b = np.power(circle_calc[1]-circle_seen[1], 2)
                    distance = np.sqrt(a + b)
                    if distance > dist_to_circ:
                        continue
                    circle_x = circle_seen[0]
                    circle_y = circle_seen[1]
                    radius = circle_seen[2]
                    depth = 0
                    color = (255, 0, 0)
                    new_pattern[row][column] = [circle_x, circle_y, radius, color, depth]
                    break
        return new_pattern

    def __get_avg_depth(self, pattern, depth_frame):
        depth_sum = 0
        depth_div = 0
        for row in range(len(pattern)):
            for column in range(len(pattern[0])):
                x, y = pattern[row][column][:2]
                if x < 0 or x >= 1280 or y >= 720 or y < 0 :
                    continue
                if depth_frame.get_distance(x, y) == 0:
                    continue
                depth_sum += depth_frame.get_distance(x, y)
                depth_div += 1
        try:
            avg_depth = depth_sum/depth_div
            return avg_depth
        except ZeroDivisionError:
            return 0

    def __classify_rec_with_circ(self, name, color_image, rec, circ, depth, depth_frame, pattern):
        (w_min, w_max, h_min, h_max) = rec
        (dist, canny_limit, lower_limit, min_radius, max_radius, distance) = circ
        (z_min, z_max) = depth
        (margin1, margin2, radius) = pattern
        #color_image = self.__clahe(color_image)
        gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

        lower, upper = self.__auto_limit(gblur, 0.33)
        canny = cv.Canny(gblur, lower, upper)
        
        kernel = np.ones((5,5))
        dilated = cv.dilate(canny, kernel)
        cv.imshow('canny', canny)
        cv.imshow('dilated', dilated)
        method = cv.RETR_TREE #cv.RETR_TREE #cv.RETR_EXTERNAL
        _, contours, _ = cv.findContours(dilated, method, cv.CHAIN_APPROX_NONE)
        
        recs_with_circs = []
        for i in range(len(contours)):
            if self.__has_corners(contours[i], 4, 8) is False:
                continue
            x, y, w, h = cv.boundingRect(contours[i])
            minrect = cv.minAreaRect(contours[i])
            box = cv.boxPoints(minrect)
            box = np.int0(box)
            bottom, left, top, right = box

            center_y = (2*y+h)/2
            center_x = (2*x+w)/2

            width = np.sqrt(np.power(bottom[0]-left[0], 2)+np.power(bottom[1]-left[1], 2))
            height = np.sqrt(np.power(left[0]-top[0], 2)+np.power(left[1]-top[1], 2))

            img_zoom = color_image[y:y+h, x:x+w]

            if self.__has_lengths(width, height, w_min, w_max, h_min, h_max) is False:
                continue
            if self.__is_centered(center_x, center_y) is False:
                continue

            circles_seen = self.__has_circles(img_zoom, dist, canny_limit, lower_limit, min_radius, max_radius, x, y, )
            if circles_seen is False:
                continue

            pattern = self.__simple_pattern(box, 8, 12, margin1, margin2, radius)
            #new_pattern = self.__compare_pattern(pattern, circles_seen, distance)
            new_pattern = pattern
            avg_depth = self.__get_avg_depth(new_pattern, depth_frame)
            if avg_depth < z_min or z_max < avg_depth:
                continue
                
            print(name, width, height, avg_depth)
            recs_with_circs.append((box, contours[i], new_pattern))
            return [(box, contours[i], new_pattern)]
        return recs_with_circs

    def classify_mtp(self, color_image, depth_frame, z_depth):
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.015, 0.085, 0.124)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.015, 0.007, 0.003)
        #print('mtp_must:',a_pix, b_pix)
        name = 'mtp: '
        ofs = 0.07
        w_min = b_pix*(1-ofs)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs)
        h_max = a_pix*(1+ofs)

        ofs2 = 9
        z_min = z_depth - 0.015 - ofs2
        z_max = z_depth - 0.015 + ofs2

        # wells
        dist = 10
        canny_limit = 100
        lower_limit = 10
        min_radius = int(np.round((diameter/2.0)*0.9))
        max_radius = int(np.round((diameter/2.0)*1.1))
        radius = int(np.round(diameter/2.0))
        distance = distance * 0.9

        #print(min_radius, max_radius, radius1)
        rec = (w_min, w_max, h_min, h_max)
        circ = (dist, canny_limit, lower_limit, min_radius, max_radius, distance)
        depth = (z_min, z_max)
        pattern = (0.12, 0.13, radius)

        mtps = self.__classify_rec_with_circ(name, color_image, rec, circ, depth, depth_frame, pattern)
        return mtps
  
    def classify_box_yellow(self, color_image, depth_frame, z_depth):   
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.05, 0.09, 0.12)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.075, 0.006, 0.003)
        #print('box_must:',a_pix, b_pix)
        name = 'box_yellow: '

        ofs = 0.07
        w_min = b_pix*(1-ofs*2/3)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs*2/3)
        h_max = a_pix*(1+ofs)
        #670/500 710/520
        ofs2 = 9
        z_min = z_depth - 0.045 - ofs2
        z_max = z_depth - 0.045 + ofs2

        # wells
        dist = 10
        canny_limit = 50
        lower_limit = 10
        min_radius = int(np.round((diameter/2.0)*0.9))
        max_radius = int(np.round((diameter/2.0)*1.1))
        radius = int(np.round(diameter/2.0))
        distance = distance * 0.9

        rec = (w_min, w_max, h_min, h_max)
        circ = (dist, canny_limit, lower_limit, min_radius, max_radius, distance)
        depth = (z_min, z_max)
        #margin = (0.07, 0.09)
        pattern = (0.10, 0.15, radius)

        boxes_yellow = self.__classify_rec_with_circ(name, color_image, rec, circ, depth, depth_frame, pattern)
        return boxes_yellow

    def classify_box_blue(self, color_image, depth_frame, z_depth):   
        #TODO Parameter anpassen
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.075, 0.09, 0.12)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.075, 0.006, 0.003)
        #print('box_must:',a_pix, b_pix)
        name = 'box_blue: '

        ofs = 0.05
        w_min = b_pix*(1-ofs*2/3)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs*2/3)
        h_max = a_pix*(1+ofs)
        #670/500 710/520
        ofs2 = 9
        z_min = z_depth - 0.045 - ofs2
        z_max = z_depth - 0.045 + ofs2

        # wells
        dist = 10
        canny_limit = 50
        lower_limit = 10
        min_radius = int(np.round((diameter/2.0)*0.9))
        max_radius = int(np.round((diameter/2.0)*1.1))
        radius = int(np.round(diameter/2.0))
        distance = distance * 0.9

        rec = (w_min, w_max, h_min, h_max)
        circ = (dist, canny_limit, lower_limit, min_radius, max_radius, distance)
        depth = (z_min, z_max)
        #margin = (0.07, 0.09)
        pattern = (0.10, 0.15, radius)

        boxes_yellow = self.__classify_rec_with_circ(name, color_image, rec, circ, depth, depth_frame, pattern)
        return boxes_yellow

    def classify_phenol_color(self, color_image, z_depth):
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.01, 0.06, 0.06)
        bound_area = a_pix * b_pix
        a_area = (np.pi * a_pix**2)/ 4
        #print('phenol_must:', bound_area,a_area)
        binar = self.__get_binar(color_image,[0 ,37,168,112,148,255])
        cv.imshow('binar_red', binar)
        kernel = np.ones((17,17))
        dilated = cv.dilate(binar, kernel)
        _,contours,_ = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            x,y,w,h = cv.boundingRect(cnt)
            bounding_area = w*h
            
            if bound_area*0.4 < bounding_area < bound_area*1.25 and a_area*0.4 < area < a_area*1.25 :
                if 0.75 * h < w < 1.25 * h:
                    print('phenol: ', bounding_area, area)
                    return (cnt, (255,255,0))

    def check_tips_depth(self, circles, depth_frame, depth_min, depth_max):
        new_circles = []
        for row in range(len(circles)):
            new_circles.append([])
            for column in range(len(circles[0])):
                depth = 0.0
                circle = circles[row][column]
                x,y = circle[:2]
                div = 0
                for i in range(-2,3):
                    for j in range(-2,3):
                        if 0 <= x <= 1280 and 0 <= y <=720:
                            depth += depth_frame.get_distance(x+j,y+i)
                            div += 1
                if div == 0:
                    div = 1
                depth /= (0.01*div)
                new_circle = circle[:]
                new_circle[4] = depth
                if depth_max > depth > depth_min:
                    new_circle[3] = (255,0,0)
                new_circles[row].append(new_circle)
        return new_circles

    def check_tips_color(self, circles, img, blue_min, green_min, red_min):
        new_circles = []
        for row in range(len(circles)):
            new_circles.append([])
            for column in range(len(circles[0])):
                circle = circles[row][column]
                x,y = circle[:2]
                blue = 0 
                green = 0
                red = 0
                div = 0
                for i in range(-2,3):
                    for j in range(-2,3):
                        if 720 > y+i >= 0 and 1280 > x+j >= 0:
                            (b,g,r) = img[y+i][x+j]
                            blue += b
                            green += g
                            red += r
                            div += 1
                blue /= div
                green /= div
                red /= div
                new_circle = circle[:]
                if blue >= blue_min and green >= green_min and red >= red_min:
                    new_circle[3] = (255,0,0)
                new_circles[row].append(new_circle)
        return new_circles

import rospy
if __name__ == "__main__":
    #rospy.init_node('classifier_node')
    my_camera = camera.Camera2()
    my_classifier = Classifier()
    my_painter = painter.Painter()
    time.sleep(1)

    while True:
        (ORIGINAL, DEPTH_FRAME) = my_camera.get_images()
        #x,y,z = my_camera.get_position()
        z = 0.2
        # ORIGINAL = cv.imread('/home/mike/training/batch_2/training_image65.jpeg')
        # scaledarray = (ORIGINAL_DEPTH_IMAGE/np.max(ORIGINAL_DEPTH_IMAGE))*255

        image_drawn = ORIGINAL.copy()
        mtps = my_classifier.classify_mtp(ORIGINAL, DEPTH_FRAME, z + 0.1)
        #boxes_blue = my_classifier.classify_box_blue(ORIGINAL, DEPTH_FRAME, z + 0.1)
        boxes_yellow = my_classifier.classify_box_yellow(ORIGINAL, DEPTH_FRAME, z + 0.1)
        phenol = my_classifier.classify_phenol_color(ORIGINAL, z + 0.1)
        ofs_left = 7

        image_drawn = my_painter.draw_mtp(mtps, image_drawn, ofs_left)
        image_drawn = my_painter.draw_box(boxes_yellow, image_drawn, ofs_left)
        #image_drawn = my_painter.draw_box(boxes_blue, image_drawn, ofs_left)
        image_drawn = my_painter.draw_phenol(phenol, image_drawn, ofs_left)

        cv.imshow('ORIGINAL', ORIGINAL)
        cv.imshow('image_drawn', image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break