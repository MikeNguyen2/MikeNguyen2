import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

ORIGINAL = cv.imread('/home/mike/cobot/scripts/classes/MTP/lab2.png')
image = ORIGINAL.copy()

gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

lower, upper = auto_limit(gblur, 0.33)
canny = cv.Canny(image, lower, upper)
canny = cv.dilate(canny, (39,39))

cv.imshow("canny", canny)

_,contours,_ = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
final = image.copy()

for contour in contours:
    area  = cv.contourArea(contour)
    length = len(contour)
    approx = cv.approxPolyDP(contour, 0.12*length, False)
    number_of_corners = len(approx)
    if 108000 > area > 800:
        final = cv.drawContours(final,contour, -1, (0,255,0), 3)
        print(area)
        x,y,w,h = cv.boundingRect(contour)

cv.imshow("final", final)
cv.waitKey(0)

# matching template with points
def distance(point_a, point_b):
    diff_x = np.absolute(point_a[0]-point_b[0])
    diff_y = np.absolute(point_a[1]-point_b[1])
    return np.sqrt(np.power(diff_x,2)+np.power(diff_y,2))
    # c = sqrt(|a1-b1|^2+|a2-b2|^2)

# doesnt work -> for the idea
tips = [[0,0]*12]*8
circle_centers = [[0,0]*12]*8
for point in tips:
    for circle in circle_centers:
        distance = distance(point,circle)
        if distance > 10: continue
        point = circle
