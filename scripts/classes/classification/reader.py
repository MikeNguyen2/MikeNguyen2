import cv2 as cv
import numpy as np


def draw_outer():
    ORIGINAL = cv.imread("/home/mike/cobot/scripts/classes/MTP/reader_front.png")

    image = ORIGINAL.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    sigma = 0.33
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))

    canny = cv.Canny(image,lower_limit,upper_limit)

    cv.imshow("original",ORIGINAL)
    cv.imshow("canny", canny)
    
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20,
                                param1=upper_limit, param2=15,
                                minRadius=80, maxRadius=100)#70 90

    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 255), 3)

    circles_small = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 50,
                                param1=upper_limit, param2=20,
                                minRadius=20, maxRadius=80)#70 90

    print(circles_small)
    if circles_small  is not None:
        circles_small  = np.uint16(np.around(circles_small ))
        for i in circles_small [0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 255, 255), 3)

    cv.imshow("circles", image)

def draw_inner():
    ORIGINAL = cv.imread("/home/mike/cobot/scripts/classes/MTP/reader_inner_pos.png")

    image = ORIGINAL.copy()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    sigma = 0.33
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))

    canny = cv.Canny(image,lower_limit,upper_limit)

    cv.imshow("original",ORIGINAL)
    cv.imshow("canny", canny)
    
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 25,
                                param1=upper_limit, param2=18,
                                minRadius=20, maxRadius=30)#70 90

    print(circles)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv.circle(image, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv.circle(image, center, radius, (255, 0, 255), 3)
    cv.imshow("circles", image)

if __name__ == "__main__":
    draw_outer()
    #draw_inner()
    cv.waitKey(0)