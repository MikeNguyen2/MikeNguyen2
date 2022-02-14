import cv2 as cv
import numpy as np

class Painter():
    def __init__(self):
        self.offset_top = 40
        self.offset_left = 7
    
    def increase_angle(self, points):
        result = 0
        x1 = np.power(points[0][0]-points[1][0], 2)
        y1 = np.power(points[0][1]-points[1][1], 2)
        x2 = np.power(points[1][0]-points[2][0], 2)
        y2 = np.power(points[1][1]-points[2][1], 2)
        length_a = np.sqrt(x1 + y1)
        length_b = np.sqrt(x2 + y2)

        if length_a > length_b:
            result += 90

        
        return result

    def draw_rec_with_circ(self, image, contour, circles, name):
        # image = cv.drawContours(image, contour, -1, (255,255,255), 3)
        minrect = cv.minAreaRect(contour)
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        x,y,w,h = cv.boundingRect(box)
        
        image = cv.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 3)
        image = cv.putText(image, name, (x,y), cv.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 3, cv.LINE_AA)

        for i in range(4):
            cv.line(image, (box[i][0], box[i][1]), (box[i-1][0], box[i-1][1]), (255, 255, 0), 2, cv.LINE_AA)
        
        for row in range(len(circles)):
            for column in range(len(circles[0])):
                circle = circles[row][column]
                # cv.circle(image, (circle[0], circle[1]), circle[2], circle[3], 2)
                cv.circle(image, (circle[0], circle[1]), circle[2], circle[3], 2)
                # image = cv.putText(image, str(int(circle[4])), (circle[0],circle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 1, cv.LINE_AA)
                # image = cv.putText(image, str(int(circle[2])), (circle[0],circle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 1, cv.LINE_AA)
        return image

    def draw_rec(self, image, contour, circles, name):
        # image = cv.drawContours(image, contour, -1, (255,255,255), 3)
        minrect = cv.minAreaRect(contour)
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        x,y,w,h = cv.boundingRect(box)

        image = cv.rectangle(image, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 3)
        image = cv.putText(image, name, (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3, cv.LINE_AA)

        for i in range(4):
            cv.line(image, (box[i][0], box[i][1]), (box[i-1][0], box[i-1][1]), (255, 255, 0), 2, cv.LINE_AA)
        return image

    def draw_mtp(self, mtps, image_drawn):
        if len(mtps)>0:
            for mtp in mtps:
                (box_points, contour, pattern) = mtp
                x, y, w, h = cv.boundingRect(box_points)
                ((center_x, center_y), _, mtp_angle) = cv.minAreaRect(box_points) 
                mtp_angle *= -1
                increase = self.increase_angle(box_points)
                mtp_angle += increase

                image_drawn = self.draw_rec_with_circ(image_drawn, contour, pattern, 'Mtp')
                image_drawn = cv.putText(image_drawn, "object: MTP", (self.offset_left,20), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)
                image_drawn = cv.putText(image_drawn, "position: " + str(int(center_x)) + ", " + str(int(center_y)), (self.offset_left,40), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                image_drawn = cv.putText(image_drawn, "angle: " + str(int(mtp_angle)), (self.offset_left,60), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)
        else:
            image_drawn = cv.putText(image_drawn, "MTP: None", (self.offset_left,20), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv.LINE_AA)
        
        return image_drawn

    def draw_box(self, boxes_yellow, image_drawn):
        if len(boxes_yellow) > 0:
            for box_yellow in boxes_yellow:
                (box_points, contour, pattern) = box_yellow
                x, y, w, h = cv.boundingRect(box_points)
                ((center_x, center_y), _, yellow_angle) = cv.minAreaRect(box_points) 
                yellow_angle *= -1
                increase = self.increase_angle(box_points)
                yellow_angle += increase

                image_drawn = self.draw_rec_with_circ(image_drawn, contour, pattern, 'Tip-Rack')
                image_drawn = cv.putText(image_drawn, "object: Tip-Rack", (self.offset_left,100), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)
                image_drawn = cv.putText(image_drawn, "position: " + str(int(center_x)) + ", " + str(int(center_y)), (self.offset_left,120), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
                image_drawn = cv.putText(image_drawn, "angle: " + str(int(yellow_angle)), (self.offset_left,140), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 255, 0), 1, cv.LINE_AA)    
        else:
            image_drawn = cv.putText(image_drawn, "Tip-Rect: None", (self.offset_left,100), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv.LINE_AA)
        return image_drawn

    def draw_phenol(self, phenol, image_drawn):
        if phenol is not None:
            (contour, color) = phenol
            image_drawn = cv.drawContours(image_drawn, contour, -1, color, 3)
            x,y,w,h = cv.boundingRect(contour)
            center_x = x + w/2
            center_y = y + h/2
            image_drawn = cv.drawContours(image_drawn, contour, -1, color, 3)
            image_drawn = cv.rectangle(image_drawn, (x-20,y-20), (x+w+20,y+h+20), (0,0,255), 3)
            image_drawn = cv.putText(image_drawn, 'Phenol', (x,y), cv.FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 3, cv.LINE_AA)
            image_drawn = cv.putText(image_drawn, "object: Phenol", (self.offset_left, 180), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
            image_drawn = cv.putText(image_drawn, "position: " + str(int(center_x)) + ", " + str(int(center_y)), (self.offset_left,200), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv.LINE_AA)
        else: 
            image_drawn = cv.putText(image_drawn, "Phenol: None", (self.offset_left, 180), cv.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 1, cv.LINE_AA)
        return image_drawn