import cv2 as cv
import numpy as np
import items


class Classifier():
    def __init__(self):
        pass

    def get_contours(self, color_image):
        color_image = color_image
        gray = cv.cvtColor(color_image, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (3, 3), cv.BORDER_DEFAULT)

        lower, upper = self.__auto_limit(gblur, 0.33)
        adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        kernel = np.ones((3,3))
        dilated = cv.dilate(adaptive, kernel)
        opening = cv.erode(dilated, kernel)
        eroded = cv.erode(opening, kernel)
        closing = cv.dilate(eroded, kernel)
        canny = cv.Canny(closing, lower, upper)

        method = cv.RETR_TREE #cv.RETR_TREE #cv.RETR_EXTERNAL
        _, contours, _ = cv.findContours(canny, method, cv.CHAIN_APPROX_NONE)
        return contours


    def classify_rec_with_circ(self, object):
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


    def classify_mtp(self):
        pass

    def classify_tablet(self):
        pass

if __name__ == "__main__":
    my_classifier = Classifier()
    mtp_data = items.Microplate()
    mtp_image = my_classifier.classify_rec_with_circ(mtp_data)