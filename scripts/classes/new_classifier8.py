from numpy.lib.shape_base import apply_over_axes
import camera
import painter
import cv2 as cv
import numpy as np
import new_painter as painter
import time

class Classifier:

    def __init__(self):
        """
        initializes the classifier object.
        """
        self.hi = 'hi'
        self.color_image = None
        self.depth_frame = None
        self.contours = None

    def __get_binar(self, spectrum):
        """
        returns a binary image of the color_image.
        Used to filter a color.
        """
        lower_limit = np.asarray(spectrum[0:3])
        upper_limit = np.asarray(spectrum[3:6])
        return cv.inRange(self.color_image,lower_limit,upper_limit) 

    def __has_corners(self, contour, conrer_min, corner_max):
        """
        returns True if the contour has the right amount of corners.
        """
        length = cv.arcLength(contour, False)
        approx = cv.approxPolyDP(contour, 0.02*length, False)
        corners = len(approx)
        if conrer_min <= corners <= corner_max:
            return True

    def __convert_height_to_pixels(self, z_depth, height_real, length_real, width_real):
        """
        Converts the real lengths to length in pixel in relation to the distance between object and camera.
        """
        #height in cm
        img_width = 1280
        img_length = 720
        # relation = 0.44 # 57.5/1280 oder 32/720 (auf Hoehe 41)
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

    def __auto_limit(self, image, sigma):
        """
        returns the best limits for the canny algorithm. 
        """
        median = np.median(image)
        lower_limit = int(max(0, (1 - sigma) * median))
        upper_limit = int(min(255, (1 + sigma) * median))
        return lower_limit, upper_limit

    def __simple_pattern(self, points, rows, columns, margin1, margin2, radius):
        """
        returns a grid with a margin for the rectangle.
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

        a_x = 50
        a_y = 0
        cv.circle(self.color_image, (a_x, a_y), 8, (255,255,0), 3, cv.LINE_AA)
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
        """
        returns True if the width and height of an object are within range.
        """
        if w_max < max(width, height) or max(width, height) < w_min:
            return False
        if h_max < min(width, height) or min(width, height) < h_min:
            return False
        return True

    def __is_centered(self, center_x, center_y):
        """
        returns True if the object is centered.
        An offset can be set 
        """
        offset = 1000
        img_cen_x = 640
        img_cen_y = 360
        if img_cen_x-offset > center_x or center_x > img_cen_x+offset:
            return False
        if img_cen_y-offset > center_y or center_y > img_cen_y+offset:
            return False
        return True

    def __has_circles(self, area, dist, canny_limit, lower_limit, min_r, max_r, x, y):
        """
        returns the cicles found by opencv (when the number of circles reaches a threshold)
        """
        (x, y, w, h) = area
        img_zoom = self.color_image[y:y+h, x:x+w]
        gray = cv.cvtColor(img_zoom, cv.COLOR_RGB2GRAY)
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
        """
        returns the pattern after readjusting the position of the wells with the seen circles.
        """
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

    def __get_avg_depth(self, pattern):
        """
        returns the average depth of the pattern.
        0 represents an unkown depth
        """
        depth_sum = 0
        depth_div = 0
        for row in range(len(pattern)):
            for column in range(len(pattern[0])):
                x, y = pattern[row][column][:2]
                if x < 0 or x >= 1280 or y >= 720 or y < 0 :
                    continue
                depth = self.depth_frame.get_distance(x, y)
                if depth == 0 or depth > 1:
                    continue
                depth_sum += depth
                depth_div += 1
        try:
            avg_depth = depth_sum/depth_div
            return avg_depth
        except ZeroDivisionError:
            return 0

    def __classify_rec_with_circ(self, name, rec, circ, depth, pattern):
        """
        returns the rectangle(s) in the image.
        by commenting the second to last line, all rectangles are returned.
        """
        (w_min, w_max, h_min, h_max) = rec
        (dist, canny_limit, lower_limit, min_radius, max_radius, distance) = circ
        (z_min, z_max) = depth
        (margin1, margin2, radius) = pattern
        
        recs_with_circs = []
        for i in range(len(self.contours)):
            if self.__has_corners(self.contours[i], 4, 8) is False:
                continue
            x, y, w, h = cv.boundingRect(self.contours[i])
            minrect = cv.minAreaRect(self.contours[i])
            box = cv.boxPoints(minrect)
            box = np.int0(box)
            bottom, left, top, right = box

            center_y = (2*y+h)/2
            center_x = (2*x+w)/2

            width = np.sqrt(np.power(bottom[0]-left[0], 2)+np.power(bottom[1]-left[1], 2))
            height = np.sqrt(np.power(left[0]-top[0], 2)+np.power(left[1]-top[1], 2))

            if self.__has_lengths(width, height, w_min, w_max, h_min, h_max) is False:
                continue
            if self.__is_centered(center_x, center_y) is False:
                continue
            circles_seen = self.__has_circles((x,y,w,h), dist, canny_limit, lower_limit, min_radius, max_radius, x, y)
            if circles_seen is False:
                continue

            pattern = self.__simple_pattern(box, 8, 12, margin1, margin2, radius)
            #new_pattern = self.__compare_pattern(pattern, circles_seen, distance)
            new_pattern = pattern
            avg_depth = self.__get_avg_depth(new_pattern)
            print(avg_depth, z_min, z_max)
            if avg_depth < z_min or z_max < avg_depth:
                continue

            new_pattern = self.check_tips_depth(pattern, z_min, z_max)
            #new_pattern = self.check_tips_color(pattern, 40, 40, 40)#40 90 190

            print(name, width, height, avg_depth)
            print(w_min, w_max, h_min, h_max)
             	
            ((center_x, center_y), (width, height), angle) = cv.minAreaRect(self.contours[i])
            rect = cv.minAreaRect(self.contours[i])
            x,y,w,h = cv.boundingRect(self.contours[i])
            box = cv.boxPoints(rect)
            box = np.int0(box)

            x1 = np.power(box[0][0]-box[1][0], 2)
            y1 = np.power(box[0][1]-box[1][1], 2)
            x2 = np.power(box[1][0]-box[2][0], 2)
            y2 = np.power(box[1][1]-box[2][1], 2)
            length_a = np.sqrt(x1 + y1)
            length_b = np.sqrt(x2 + y2)
            x_rotate = box[1][0]
            y_rotate = box[1][1]

            if length_a > length_b:
                angle += 90
                x_rotate = box[2][0]
                y_rotate = box[2][1]
            gray = cv.cvtColor(self.color_image, cv.COLOR_RGB2GRAY)
            gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)
            adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
            
            h_rotate = int(min(length_a,length_b))
            w_rotate = int(max(length_a,length_b))

            image_center = (x_rotate,y_rotate)#tuple(np.array(self.color_image.shape[1::-1]) / 2)
            rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
            color_rotate = cv.warpAffine(self.color_image, rot_mat, self.color_image.shape[1::-1], flags=cv.INTER_LINEAR)
            adaptive_rotate = cv.warpAffine(adaptive, rot_mat, adaptive.shape[1::-1], flags=cv.INTER_LINEAR)
           
            #print(x_rotate, y_rotate, h_rotate, w_rotate)
            adap_zoom = adaptive_rotate[y_rotate:y_rotate+h_rotate, x_rotate:x_rotate+w_rotate]
            color_zoom = color_rotate[y_rotate:y_rotate+h_rotate, x_rotate:x_rotate+w_rotate].copy()
            m2 = int(margin2*h_rotate*0.7)
            m1 = int(margin1*w_rotate*0.7)

            top = adap_zoom[:m2,:]
            bottom = adap_zoom[h_rotate-m2:,:]
            left = adap_zoom[:,:m1]
            right = adap_zoom[:,w_rotate-m1:]

            color_left = color_zoom[:,:m1]
            _,contours_left,_ = cv.findContours(left,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

            cn_left = 0
            cn_right = 0

            for contour_left in contours_left:
                contour_area = cv.contourArea(contour_left)
                if contour_area > 30:
                    color_left = cv.drawContours(color_left, contour_left, -1, (128,255,0), 1)
                    cn_left += 1

            color_right = color_zoom[:,w_rotate-m1:]
            _,contours_right,_ = cv.findContours(right,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)

            for contour_right in contours_right:
                contour_area = cv.contourArea(contour_right)
                if contour_area > 30:
                    color_right = cv.drawContours(color_right, contour_right, -1, (128,255,0), 1)
                    cn_right +=1

            if cn_right > cn_left and cn_right > 6: print(180)
            else: print(0)

            cv.circle(color_rotate, (x_rotate, y_rotate), 3, (128,128,128), 7)
            cv.circle(color_rotate, (x_rotate+w_rotate, y_rotate), 3, (128,128,128), 7)
            cv.circle(color_rotate, (x_rotate, y_rotate+h_rotate), 3, (128,128,128), 7)
            #cv.imshow('adaptive_rotate', adaptive_rotate)
            #cv.imshow('adap_zoom', adap_zoom)
            #cv.imshow('rotate', color_rotate)
            #cv.imshow('color_zoom', color_zoom)

            recs_with_circs.append((box, self.contours[i], new_pattern))
            return [(box, self.contours[i], new_pattern)]
        return recs_with_circs

    def classify_mtp(self, z_depth):
        """
        returns a list of mtps.
        """
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.015, 0.08, 0.124)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.015, 0.007, 0.003)
        #print('mtp_must:',a_pix, b_pix)
        name = 'mtp: '
        ofs = 0.07
        w_min = b_pix*(1-ofs)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs)
        h_max = a_pix*(1+ofs)

        ofs2 = 0.01
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
        pattern = (0.1, 0.115, radius)

        mtps = self.__classify_rec_with_circ(name, rec, circ, depth, pattern)
        return mtps
  
    def classify_box_yellow(self, z_depth):   
        """
        returns a list of yellow boxes.
        """
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.05, 0.085, 0.124)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.05, 0.006, 0.003) 
        #print('box_must:',a_pix, b_pix)
        name = 'box_yellow: '

        ofs = 0.02
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

        boxes_yellow = self.__classify_rec_with_circ(name, rec, circ, depth, pattern)
        return boxes_yellow

    def classify_box_orange(self, z_depth):   
        """
        returns a list of orange boxes.
        """
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.035, 0.08, 0.112)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.035, 0.005, 0.003) 
        name = 'box_orange: '

        ofs = 0.15
        w_min = b_pix*(1-ofs*2/3)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs*2/3)
        h_max = a_pix*(1+ofs)
        
        ofs2 = 0.01
        z_tips = 0.015
        z_min = z_depth - 0.045 - ofs2 - z_tips
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
        pattern = (0.0725, 0.11, radius)

        boxes_yellow = self.__classify_rec_with_circ(name, rec, circ, depth, pattern)
        return boxes_yellow

    def classify_box_blue(self, z_depth):   
        """
        returns a list of blue boxes.
        """
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.08, 0.075, 0.112)
        (diameter, distance) = self.__convert_height_to_pixels(z_depth, 0.08, 0.008, 0.003)
        
        #print('box_must:',a_pix, b_pix)
        name = 'box_blue: '

        ofs = 0.02#0.02
        w_min = b_pix*(1-ofs*2/3)
        w_max = b_pix*(1+ofs)
        h_min = a_pix*(1-ofs*2/3)
        h_max = a_pix*(1+ofs)
        print(w_min, w_max, h_min, h_max, "box")
        #670/500 710/520
        ofs2 = 9
        z_min = z_depth - 0.08 - ofs2
        z_max = z_depth - 0.08 + ofs2

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

        pattern = (0.07, 0.11, radius)

        boxes_blue = self.__classify_rec_with_circ(name, rec, circ, depth, pattern)
        return boxes_blue

    def classify_phenol_color(self, z_depth):
        """
        returns the phenol.
        not in a list, because probably only 1 at a time.
        """
        (a_pix, b_pix) = self.__convert_height_to_pixels(z_depth, 0.01, 0.06, 0.06)
        bound_area = a_pix * b_pix
        a_area = (np.pi * a_pix**2)/ 4
        #print('phenol_must:', bound_area,a_area)
        binar = self.__get_binar([0 ,37,168,112,148,255])
        cv.imshow('binar_red', binar)
        kernel = np.ones((17,17))
        dilated = cv.dilate(binar, kernel)
        _,contours,_ = cv.findContours(dilated,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
        for cnt in contours:
            area = cv.contourArea(cnt)
            x,y,w,h = cv.boundingRect(cnt)
            bounding_area = w*h
            
            if bound_area*0.4 < bounding_area < bound_area*1.25 and a_area*0.4 < area < a_area*1.25 :
                if 0.85 * h < w < 1.15 * h:
                    print('phenol: ', bounding_area, area)
                    return (cnt, (255,255,0))

    def check_tips_depth(self, circles, depth_min, depth_max):
        """
        returns the pattern, after changing the color of present tips.
        using the depth, it guesses if the tip is present or not
        """
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
                        if 0 < x+j < 1280 and 0 < y+i < 720:
                            depth += self.depth_frame.get_distance(x+j,y+i)
                            div += 1
                if div == 0: div = 1
                depth /= (div)
                new_circle = circle[:]
                new_circle[4] = depth
                if depth == 0: pass
                elif depth_min-0.015 < depth < depth_min + 0.015:
                    new_circle[3] = (0,0,0)
                else:
                    new_circle[3] = (255,0,0)
                new_circles[row].append(new_circle)
        return new_circles

    def check_tips_color(self, circles, blue_min, green_min, red_min):
        """
        returns the pattern, after changing the color of present tips.
        using color, it guesses if the tip is present or not. 
        (usable because of the contrast between black(not) and white(present))
        """
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
                            (b,g,r) = self.color_image[y+i][x+j]
                            blue += b
                            green += g
                            red += r
                            div += 1
                try:
                    blue /= div
                    green /= div
                    red /= div
                except:
                    blue = 0
                    green = 0
                    red = 0
                    print('no value')
                new_circle = circle[:]
                if blue >= blue_min and green >= green_min and red >= red_min:
                    new_circle[3] = (255,0,0)
                new_circles[row].append(new_circle)
        return new_circles

    def get_contours(self, color_image):
        """
        returns the contours and shows important images.s
        may need readjustment for runtime
        """
        self.color_image = color_image
        gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (11, 11), cv.BORDER_DEFAULT)

        lower, upper = self.__auto_limit(gblur, 0.33)
        adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        kernel = np.ones((3,3))
        dilated = cv.dilate(adaptive, kernel)
        opening = cv.erode(dilated, kernel)
        eroded = cv.erode(opening, kernel)
        closing = cv.dilate(eroded, kernel)
        canny = cv.Canny(closing, lower, upper)
        canny2 = cv.Canny(gblur,lower,upper)

        cv.imshow('canny2',canny2)
        cv.imshow('canny', canny)
        cv.imshow('adaptive',adaptive)
        cv.imshow('opening', opening)
        cv.imshow('closing', closing)

        method = cv.RETR_TREE #cv.RETR_TREE #cv.RETR_EXTERNAL
        _, self.contours, _ = cv.findContours(canny, method, cv.CHAIN_APPROX_NONE)
        return self.contours

    def set_images(self, color_image, depth_frame):
        """
        updates the current images as global variables.
        """
        self.color_image = color_image
        self.depth_frame = depth_frame

    def tablet_pattern(self, points):
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

    def get_tablet_contours(self, color_image):
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

    def classify_tablet(self, contours, z_depth):
        """
        returns the contour of the tablet and the pattern.
        """
        a_pix, b_pix = self.__convert_height_to_pixels(z_depth,0.005,0.235,0.165)
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
                circles = self.tablet_pattern(box)
                tablet_contour = contour
                break 
        return (tablet_contour, circles)

    def compare_tablet_pattern(self, contours, tablet, image_drawn, z_depth):
        """
        compares the pattern with the contours and draws on the image, then returns the image.
        letters and numbers.
        bars of the current well.
        next-button.
        """
        (tablet_contour,circles) = tablet
        if circles is None or contours is None:
            return image_drawn, None, None
        
        a_pix, b_pix = self.__convert_height_to_pixels(z_depth,0.005,0.018,0.004)
        c_pix, c_pix = self.__convert_height_to_pixels(z_depth,0.005,0.0008,0.0008)
        d_pix, e_pix = self.__convert_height_to_pixels(z_depth,0.005,0.029,0.009)

        num_letter = None
        num_number = None

        for circle in circles:
            cv.circle(image_drawn, (circle[0], circle[1]), 5, (255,0,0), 1)

        for contour in contours:
            x, y, w, h = cv.boundingRect(contour)
            minrect = cv.minAreaRect(contour)
            box = cv.boxPoints(minrect)
            box = np.int0(box)
            bottom, left, top, right = box

            width = np.sqrt(np.power(bottom[0]-left[0], 2)+np.power(bottom[1]-left[1], 2))
            height = np.sqrt(np.power(left[0]-top[0], 2)+np.power(left[1]-top[1], 2))

            if a_pix*b_pix*1.6 > width*height > a_pix*b_pix*1.2:
                cen_x = x+0.5*w
                cen_y = y+0.5*h
                for i in range(20):
                    a = np.power(circles[i][0]-cen_x, 2)
                    b = np.power(circles[i][1]-cen_y, 2)
                    
                    distance = np.sqrt(a + b)
                    if distance < 20:#4*c_pix:
                        cv.circle(image_drawn, (int(cen_x), int(cen_y)), 5, (255,0,255), 3)
                        if i <= 12:
                            num_number = i
                        else: 
                            num_letter = i - 12

        minrect = cv.minAreaRect(tablet_contour)
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
        
        cv.drawContours(image_drawn, tablet_contour, -1, (0,0,255), 1)
        cv.drawContours(image_drawn, [box], -1, (255,255,255), 1)

        try:
            let_letter = dic[num_letter]
            text = let_letter + str(num_number)
            cv.putText(image_drawn, text, (10, 720 - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv.LINE_AA)
        except:
            cv.putText(image_drawn, "unreadable", (10,720 - 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 3, cv.LINE_AA)
        return image_drawn, num_letter, num_number

if __name__ == "__main__":
    import rospy
    rospy.init_node('classifier_node')
    my_camera = camera.Camera()
    my_classifier = Classifier()
    my_painter = painter.Painter()
    time.sleep(1)

    while True:
        print
        print('objects')
        print('-----------------------------------')
        (ORIGINAL, DEPTH_FRAME) = my_camera.get_images()
        image_drawn = ORIGINAL.copy()
        x,y,z = my_camera.get_position()
        z += 0.09
        #z=0.2
        my_classifier.set_images(ORIGINAL, DEPTH_FRAME)
        contours = my_classifier.get_contours(ORIGINAL)

        mtps = my_classifier.classify_mtp(z)
        #boxes_yellow = my_classifier.classify_box_yellow(z)
        #boxes_blue = my_classifier.classify_box_blue(z)
        #phenol = my_classifier.classify_phenol_color(z)
        boxes_orange = my_classifier.classify_box_orange(z)
        tablet = my_classifier.classify_tablet(contours, z)

        image_drawn = my_painter.draw_mtp(mtps, image_drawn)
        #image_drawn = my_painter.draw_box(boxes_yellow, image_drawn, ofs_left)
        #image_drawn = my_painter.draw_mtp(boxes_blue, image_drawn, ofs_left)
        image_drawn = my_painter.draw_box(boxes_orange, image_drawn)
        #image_drawn = my_painter.draw_phenol(phenol, image_drawn, ofs_left)
        image_drawn,_,_ = my_classifier.compare_tablet_pattern(contours, tablet, image_drawn, z)
        cv.imshow('ORIGINAL', ORIGINAL)
        cv.imshow('image_drawn', image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

        