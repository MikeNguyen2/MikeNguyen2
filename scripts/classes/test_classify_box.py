import camera
import painter
import cv2 as cv
import numpy as np
import random

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

if __name__ ==  "__main__":
    my_camera = camera.Camera()
    my_painter = painter.Painter()

    while True:
        ORIGINAL = my_camera.get_image()
        image_drawn = ORIGINAL.copy()

        gray = cv.cvtColor(ORIGINAL.copy(), cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

        lower, upper = auto_limit(gblur, 0.33)
        canny = cv.Canny(gblur, lower, upper)

        dilated = cv.dilate(canny, (37,37))
        _,contours,hierachies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        for i in range(len(contours)):
            area  = cv.contourArea(contours[i])
            if 100000 > area > 40000: #and 5 >= number_of_corners >= 3:
                x,y,w,h = cv.boundingRect(contours[i])
                length = cv.arcLength(contours[i],False)
                approx = cv.approxPolyDP(contours[i],0.02*length,False)
                corners = len(approx)
                bounding_area = w*h
                
                if 40000 < bounding_area < 140000: # 4 <= corners <= 6 and
                    if hierachies[0][i][3] == -1:
                        child = hierachies[0][i][2]
                        child = hierachies[0][child][2]
                        go = True
                        cv.drawContours(image_drawn,contours[i],-1, (0,0,255),3)
                        while go and hierachies[0][child][0] != -1:
                            print(child)
                            area2 = cv.contourArea(contours[child])
                            x2,y2,w2,h2 = cv.boundingRect(contours[child])
                            bounding_area2 = w2*h2
                            if 10000 < bounding_area2:
                                cv.drawContours(image_drawn,contours[child],-1, (0,255,0),3)
                                go = False
                            else:
                                cv.drawContours(image_drawn,contours[child],-1, (0,255,0),3)
                                child = hierachies[0][child][0]


        cv.imshow('dilated', dilated)
        cv.imshow('image_drawn', image_drawn)
        cv.imshow('canny', canny)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()