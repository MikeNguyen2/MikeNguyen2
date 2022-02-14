#from typing import Pattern
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

def __mapping(value, a, b, c, d):
    return c + (d - c) * ((value - a) / float(b - a))

def __calculate_pattern( points, rows, columns, margin1, margin2):
    shortest_distance = 10**10
    closest_point = None
    random_point = random.choice(points)
    for point in points:
        if np.array_equal(random_point, point):
            continue

        distance = np.linalg.norm(random_point-point)
        if distance < shortest_distance:
            shortest_distance = distance
            closest_point = point

    pair1 = [random_point, closest_point]
    pair2 = []
    for point in points:
        if np.array_equal(pair1[0], point):
            continue
        if np.array_equal(pair1[1], point):
            continue

        pair2.append(point)

    pair1_average_x = (pair1[0][0] + pair1[1][0]) / 2.0
    pair2_average_x = (pair2[0][0] + pair2[1][0]) / 2.0

    top_left = None
    if pair1_average_x < pair2_average_x:
        if pair1[0][1] < pair1[1][1]:
            top_left = pair1[0]
        else:
            top_left = pair1[1]
    else:
        if pair2[0][1] < pair2[1][1]:
            top_left = pair2[0]
        else:
            top_left = pair2[1]

    shortest_distance = 10**10
    closest = None
    longest_distance = 0
    furthest = None
    for point in points:
        if np.array_equal(top_left, point):
            continue

        distance = np.linalg.norm(top_left-point)
        if distance < shortest_distance:
            shortest_distance = distance
            closest = point

        if distance > longest_distance:
            longest_distance = distance
            furthest = point

    closest2 = None
    for point in points:
        if np.array_equal(top_left, point):
            continue
        if np.array_equal(furthest, point):
            continue
        if np.array_equal(closest, point):
            continue
        closest2 = point

    con_closest = [closest[0]-top_left[0], closest[1]-top_left[1]]
    con_closest2 = [closest2[0]-top_left[0], closest2[1]-top_left[1]]

    um = 1-margin1
    lm = margin1
    um2 = 1-margin2
    lm2 = margin2

    positions = []
    for j in range(12):
        positions.append([])
        for i in range(8):
            well_x = top_left[0]
            well_x += __mapping(
                i, 0, rows-1, con_closest[0]*lm2, con_closest[0]*um2)
            well_x += __mapping(    
                j, 0, columns-1, con_closest2[0]*lm, con_closest2[0]*um)

            well_y = top_left[1]
            well_y += __mapping(
                i, 0, rows-1, con_closest[1]*lm2, con_closest[1]*um2)
            well_y += __mapping(
                j, 0, columns-1, con_closest2[1]*lm, con_closest2[1]*um)

            positions[j].append([np.float32(well_x), np.float32(well_y)])

    return positions

def simple_pattern(points, rows, columns, margin1, margin2):
    length_a = np.sqrt(np.power(points[0][0]-points[1][0],2)+np.power(points[0][1]-points[1][1],2))
    length_b = np.sqrt(np.power(points[1][0]-points[2][0],2)+np.power(points[1][1]-points[2][1],2))

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

    circles = []
    for row in range(rows):
        circles.append([])
        for column in range(columns):
            pos_x = top_left[0] + margin1*width1  + (width1-2*margin1*width1)  /(columns-1)*column + (width2-2*margin1*width2)/(rows-1)*row
            pos_y = top_left[1] + margin2*height2 + (height1-2*margin2*height1)/(columns-1)*column + (height2-2*margin2*height2)/(rows-1)*row
            circles[row].append([pos_x, pos_y])
    return circles

def draw_simple_pattern(image_drawn):
    length_a = np.sqrt(np.power(box[0][0]-box[1][0],2)+np.power(box[0][1]-box[1][1],2))
    length_b = np.sqrt(np.power(box[1][0]-box[2][0],2)+np.power(box[1][1]-box[2][1],2))
    
    top_left = box[1]
    top_right = box[2]
    bottom_left = box[0]
    bottom_right = box[3]

    if max(length_a,length_b) == length_a:
        top_left = box[2]
        top_right = box[3]
        bottom_left = box[1]
        bottom_right = box[0]

    image_drawn = cv.circle(image_drawn, (top_left[0],top_left[1]), 16, (0, 255, 255), 3, cv.LINE_AA)
    image_drawn = cv.circle(image_drawn, (bottom_left[0],bottom_left[1]), 16, (0, 0, 255), 3, cv.LINE_AA)

    points = [top_left,top_right,bottom_left,bottom_right]
    for i in range(4):
        image_drawn = cv.line(image_drawn, (box[i][0],box[i][1]),(box[i-1][0],box[i-1][1]), (255, 255, 0), 2, cv.LINE_AA)

    margin_short = 28.0
    margin_short_w = top_right[0]-top_left[0] + 0.0
    margin_short_h = top_right[1]-top_left[1] + 0.0
    margin_short_s = np.sqrt(np.power(margin_short_w,2) + np.power(margin_short_h,2))
    margin_short_y = (margin_short/margin_short_s)*margin_short_h
    margin_short_x = np.sqrt(np.power(margin_short,2)-np.power(margin_short_y,2))

    margin_long = 25.0
    margin_long_w = bottom_left[0]-top_left[0] + 0.0
    margin_long_h = bottom_left[1]-top_left[1] + 0.0
    margin_long_s = np.sqrt(np.power(margin_long_w,2) + np.power(margin_long_h,2))
    margin_long_y = (margin_long/margin_long_s)*margin_long_h
    margin_long_x = np.sqrt(np.power(margin_long,2)-np.power(margin_long_y,2))
    if max(length_a,length_b) == length_a:
        for i in range(12):
            start_x = top_left[0] + margin_short_x + (margin_short_w - 2*margin_short_x)/12 *i
            start_y = top_left[1] + margin_short_y + (margin_short_h - 2*margin_short_y)/12 *i
            for j in range(8):
                position_x = start_x - margin_long_x + (margin_long_w + 2*margin_long_x)/8 *j
                position_y = start_y + margin_long_y + (margin_long_h - 2*margin_long_y)/8 *j
                cv.circle(image_drawn, (int(np.round(position_x)),int(np.round(position_y))), 3, (0, 0, 255), 3, cv.LINE_AA)
    else:
        for i in range(12):
            start_x = top_left[0] + margin_short_x + (margin_short_w - 2*margin_short_x)/12 *i
            start_y = top_left[1] + margin_short_y + (margin_short_h + 2*margin_short_y)/12 *i
            for j in range(8):
                position_x = start_x + margin_long_x + (margin_long_w + 2*margin_long_x)/8 *j
                position_y = start_y + margin_long_y + (margin_long_h - 2*margin_long_y)/8 *j
                cv.circle(image_drawn, (int(np.round(position_x)),int(np.round(position_y))), 3, (0, 0, 255), 3, cv.LINE_AA)
    return image_drawn

def classify_rectangle(img, min_area, max_area):
    gray = cv.cvtColor(img.copy(), cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

    lower, upper = auto_limit(gblur, 0.33)
    canny = cv.Canny(gblur, lower, upper)

    dilated = cv.dilate(canny, (5,5))
    _,contours,_ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    for contour in contours:
        area  = cv.contourArea(contour)
        length = len(contour)
        approx = cv.approxPolyDP(contour, 0.12*length, False)
        number_of_corners = len(approx)
        if max_area > area > min_area and 5 >= number_of_corners >= 3:
            return contour

def classify_circle(img, dist, canny_limit, lower_limit ,min_radius, max_radius):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)
    dilated = cv.dilate(gblur, (5,5))

    circles = cv.HoughCircles(dilated, cv.HOUGH_GRADIENT, 1, dist,
                                param1=canny_limit, param2=lower_limit,
                                minRadius=min_radius, maxRadius=max_radius)
    return circles

def classify_rec_with_circ(img, min_area, max_area, dist, canny_limit, lower_limit ,min_radius, max_radius):
    image = img.copy()
    contour = classify_rectangle(image, min_area, max_area)
    if contour is None:
        return None
    x,y,w,h = cv.boundingRect(contour)
    circles_zoom = classify_circle(image[y:y+h,x:x+w], dist, canny_limit, lower_limit ,min_radius, max_radius)
    
    if circles_zoom is None:
        return None

    circles_seen = []
    for circle_zoom in circles_zoom[0, :]:
        new_x = circle_zoom[0] + x
        new_y = circle_zoom[1] + y
        radius = circle_zoom[2]
        circles_seen.append([new_x, new_y, radius])

    minAreaRect = cv.minAreaRect(contour)
    corner_points = cv.boxPoints(minAreaRect)
    circles_calc = simple_pattern(corner_points, 8, 12, 0.14, 0.11)

    circles = []
    for width in circles_calc:
        for circle_calc in width:
            appended = False
            for circle_seen in circles_seen:
                distance = np.sqrt(np.power(circle_calc[0]-circle_seen[0],2)+np.power(circle_calc[1]-circle_seen[1],2))
                if distance < 8:
                    x = circle_seen[0]
                    y = circle_seen[1]
                    radius = circle_seen[2]
                    color = (255, 0, 255)
                    circles.append([x, y, radius, color])
                    appended = True
                    break
            if appended == False:
                x = circle_calc[0]
                y = circle_calc[1]
                radius = 10
                color = (0, 0, 255)
                circles.append([x, y, radius, color])
    return (contour, circles)

def classify_circ_with_circ(img):
    circles = classify_circle(img, 20, 100, 25, 60, 70)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            x,x2 = circle[0] - radius, circle[0] + radius
            y,y2 = circle[1] - radius, circle[1] + radius
            img_zoom = img[y:y2,x:x2].copy()
            circles_small = classify_circle(img_zoom, 20, 100, 20, 40, 60)
            if circles_small is not None:
                return circle          

def draw_phenol(img, circle):
    center = (circle[0],circle[1])
    radius = circle[2]
    image = img.copy()
    cv.circle(image, center, 1, (0, 100, 100), 3)
    cv.circle(image, center, radius, (255, 0, 255), 3)
    return image
            
def draw_rec_with_circ(img, contour, circles):
    image = img.copy()
    image = cv.drawContours(image,contour, -1, (0,255,0), 3)
    if circles is not None:
        for circle in circles:
            circle[:2] = np.uint16(np.around(circle[:2]))
            center = (circle[0], circle[1])
            radius = circle[2]
            color = circle[3]

            cv.circle(image, center, 1, (0, 100, 100), 3)
            cv.circle(image, center, radius, color, 1)
    return image

if __name__ ==  "__main__":
    #my_camera = camera.Camera()
    my_painter = painter.Painter()

    while True:
        #ORIGINAL = my_camera.get_image()
        ORIGINAL = cv.imread("/home/mike/cobot/scripts/classes/MTP/lab.png")
        box = classify_rec_with_circ(ORIGINAL, 60000, 120000, 15, 75, 5, 10, 16)
        mtp = classify_rec_with_circ(ORIGINAL, 40000, 60000, 10, 100, 10, 8, 12)
        phenol = classify_circ_with_circ(ORIGINAL)
        image_drawn = ORIGINAL.copy()

        if phenol is not None:
            contour_phenol = phenol
            image_drawn = draw_phenol(image_drawn, contour_phenol)
        if box is not None:
            (contour_box, tips) = box
            image_drawn = draw_rec_with_circ(image_drawn, contour_box, tips)
        if mtp is not None:
            (contour_mtp, wells) = mtp
            image_drawn = draw_rec_with_circ(image_drawn, contour_mtp, wells)
        
        
        
        # image_drawn = ORIGINAL.copy()
        gray = cv.cvtColor(ORIGINAL.copy(), cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

        lower, upper = auto_limit(gblur, 0.33)
        canny = cv.Canny(gblur, lower, upper)

        # dilated = cv.dilate(canny, (37,37))
        # _,contours,_ = cv.findContours(dilated, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # for contour in contours:
        #     area  = cv.contourArea(contour)
        #     length = len(contour)
        #     approx = cv.approxPolyDP(contour, 0.12*length, False)
        #     number_of_corners = len(approx)
        #     if area > 10000 and 4 >= number_of_corners >= 3:
        #         cv.drawContours(image_drawn,contour, -1, (0, 255, 0), 1)
            
        # cv.imshow('dilated', dilated)
        cv.imshow('image_drawn', image_drawn)
        cv.imshow('canny', canny)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()
    
