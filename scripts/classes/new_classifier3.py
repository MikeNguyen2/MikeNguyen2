import camera
import painter
import cv2 as cv
import numpy as np

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

def draw_pattern(circles, image):
    for row in range(len(circles)):
        for column in range(len(circles[0])):
            circle = circles[row][column]
            cv.circle(image,(circle[0],circle[1]),circle[2],circle[3],2)
    return image

def find_rect(w_max,w_min,y_max,y_min,color,contours):
    for i in range(len(contours)):
        x,y,w,h = cv.boundingRect(contours[i])
        minrect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        a,b,c,d = box
        w = np.sqrt(np.power(a[0]-b[0],2)+np.power(a[1]-b[1],2))
        h = np.sqrt(np.power(b[0]-c[0],2)+np.power(b[1]-c[1],2))
        if w_max > max(w,h) > w_min and y_max > min(w,h) > y_min:
            center_y = (2*y+h)/2
            center_x = (2*x+w)/2
            offset = 1000
            img_cen_x = 640
            img_cen_y = 360
            if img_cen_x-offset < center_x < img_cen_x+offset and img_cen_y-offset< center_y < img_cen_y+offset:
                length = cv.arcLength(contours[i],False)
                approx = cv.approxPolyDP(contours[i],0.02*length,False)
                corners = len(approx)
                cv.drawContours(image_drawn,contours[i],-1, color,3)
                if 4 <= corners <= 6:
                    cv.drawContours(image_drawn,contours[i],-1, (255,0,255),3)
                    #cv.rectangle(image_drawn, (x,y), (x+w,y+h),(0,255,0),3)
                return (box, cv.boundingRect(contours[i]))
    return None

def find_circles(bounding, img, dist, canny_limit, lower_limit ,min_radius, max_radius):
    x,y,w,h = bounding
    img_zoom = img[y:y+h,x:x+w]
    gray = cv.cvtColor(img_zoom, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)
    dilated = cv.dilate(gblur, (5,5))

    circles_zoom = cv.HoughCircles(dilated, cv.HOUGH_GRADIENT, 1, dist,
                                param1=canny_limit, param2=lower_limit,
                                minRadius=min_radius, maxRadius=max_radius)
    circles = []
    if circles_zoom is not None:
        for circle_zoom in circles_zoom[0]:
            circle_new = (int(circle_zoom[0]+x), int(circle_zoom[1]+y), int(circle_zoom[2]))
            circles.append(circle_new)
    return circles

def simple_pattern(points, rows, columns, margin1, margin2):
    length_a = np.sqrt(np.power(points[0][0]-points[1][0],2)+np.power(points[0][1]-points[1][1],2))
    length_b = np.sqrt(np.power(points[1][0]-points[2][0],2)+np.power(points[1][1]-points[2][1],2))
    for i in range(4):
        cv.line(image_drawn, (points[i][0],points[i][1]),(points[i-1][0],points[i-1][1]), (255, 255, 0), 2, cv.LINE_AA)

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

    top_form = np.power(margin1,2)*(np.power(width1,2)+np.power(height1,2))
    bottom_form = 1 + np.power(height1/float(width1),2)
    margin_x = np.sqrt(top_form/bottom_form)
    margin_y = margin_x * height1/float(width1)

    top_form2 = np.power(margin2,2)*(np.power(width2,2)+np.power(height2,2))
    bottom_form2 = 1 + np.power(height2/float(width2),2)
    margin_x2 = np.sqrt(top_form2/bottom_form2)
    margin_y2 = margin_x2 * height2/float(width2)

    circles = []
    if length_a > length_b:
        for column in range(columns):
            circles.append([])
            start_x = top_left[0] + margin_x + (width1 - 2*margin_x)/(columns-1)*column 
            start_y = top_left[1] + margin_y + (height1- 2*margin_y)/(columns-1)*column 
            
            for row in range(rows):
                pos_x = start_x - margin_x2 - (width2 - 2*margin_x2)/(rows-1)*row 
                pos_y = start_y + margin_y2 + (height2 - 2*margin_y2)/(rows-1)*row 
                circles[column].append([int(pos_x), int(pos_y),5,(255,255,255)])  
    else:      
        for column in range(columns):
            circles.append([])
            start_x = top_left[0] + margin_x + (width1 - 2*margin_x)/(columns-1)*column 
            start_y = top_left[1] - margin_y - (height1 - 2*margin_y)/(columns-1)*column 
            
            for row in range(rows):
                pos_x = start_x + margin_x2 + (width2 - 2*margin_x2)/(rows-1)*row 
                pos_y = start_y + margin_y2 + (height2- 2*margin_y2)/(rows-1)*row 
                circles[column].append([int(pos_x), int(pos_y),5,(255,255,255)]) 
                       
                       
    return circles

def compare_pattern(pattern, circles_seen):
    new_pattern = pattern[:]
    for row in range(len(pattern)):
        for column in range(len(pattern[0])):
            circle_calc = pattern[row][column]
            for circle_seen in circles_seen:
                distance = np.sqrt(np.power(circle_calc[0]-circle_seen[0],2)+np.power(circle_calc[1]-circle_seen[1],2))
                if distance < 6:
                    x = circle_seen[0]
                    y = circle_seen[1]
                    radius = circle_seen[2]
                    color = (255, 0, 255)
                    new_pattern[row][column] = [x, y, radius, color]
                    break
    return new_pattern

def classify_mtp(contours,depth_image):
    ofs = 0
    mtp = find_rect(290+ofs,260-ofs,200+ofs,175-ofs,(255,0,0),contours,0.3,0.35,depth_image)
    if mtp is not None:
        (rect_mtp, bound_mtp) = mtp
        pattern_mtp = simple_pattern(rect_mtp,8,12,0.11,0.13)
        circles_mtp = find_circles(bound_mtp, ORIGINAL.copy(), 10, 100, 10, 8, 12)
        new_pattern_mtp = compare_pattern(pattern_mtp, circles_mtp)
        return (mtp,new_pattern_mtp)

def classify_box(contours,depth_image):
    ofs = 0
    box = find_rect(325+2*ofs,280+ofs,235+2*ofs,190,(0,0,2-5),contours,0.4,0.45,depth_image)
    if box is not None:
        (rect_box, bound_box) = box
        pattern_box = simple_pattern(rect_box, 8,12,0.11,0.14)
        circles_box = find_circles(bound_box, ORIGINAL.copy(), 15, 75, 5, 10, 16)
        new_pattern_box = compare_pattern(pattern_box, circles_box)
        return (box,new_pattern_box)

def classify_phenol(img):
    bounding = (0,0,1280,720)
    circles = find_circles(bounding, img, 20, 100, 25, 60, 70)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            x,x2 = circle[0] - radius, circle[0] + radius
            y,y2 = circle[1] - radius, circle[1] + radius
            bounding = (x,y,2*radius,2*radius)
            circles_small = find_circles(bounding, img, 20, 100, 20, 40, 60)
            if circles_small is not None:
                return (circle[0],circle[1],circle[2],(0,0,255))

if __name__ ==  "__main__":
    my_camera = camera.Camera()
    my_painter = painter.Painter()

    while True:
        # ORIGINAL = my_camera.get_image()
        # ORIGINAL = cv.imread("/home/mike/cobot/scripts/classes/MTP/lab.png")
        ORIGINAL = cv.imread('/home/mike/training/batch_2/training_image65.jpeg')

        image_drawn = ORIGINAL.copy()
        gray = cv.cvtColor(ORIGINAL.copy(), cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

        lower, upper = auto_limit(gblur, 0.33)
        canny = cv.Canny(gblur, lower, upper)
        dilated = cv.dilate(canny, (37,37))

        _,contours,hierachies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        ofs = 50
        box = find_rect(325+2*ofs,280+ofs,235+2*ofs,190,(0,0,2-5),contours)
        mtp = find_rect(290+ofs,260-ofs,200+ofs,175-ofs,(255,0,0),contours)
        phenol = classify_phenol(ORIGINAL.copy())
        
        if phenol is not None:
            cv.circle(image_drawn,(phenol[0],phenol[1]),phenol[2],phenol[3],2)

        if mtp is not None:
            (rect_mtp, bound_mtp) = mtp
            pattern_mtp = simple_pattern(rect_mtp,8,12,0.11,0.13)
            circles_mtp = find_circles(bound_mtp, ORIGINAL.copy(), 1, 100, 10, 8, 12)
            new_pattern_mtp = compare_pattern(pattern_mtp, circles_mtp)
            image_drawn = draw_pattern(new_pattern_mtp, image_drawn)
            x,y,w,h = bound_mtp
            image_drawn = cv.putText(image_drawn, 'MTP', (x,y),cv.FONT_HERSHEY_SIMPLEX,1,  (255,255,0), 3, cv.LINE_AA)

        if box is not None:
            (rect_box, bound_box) = box
            pattern_box = simple_pattern(rect_box, 8,12,0.11,0.14)
            circles_box = find_circles(bound_box, ORIGINAL.copy(), 15, 75, 5, 10, 16)
            new_pattern_box = compare_pattern(pattern_box, circles_box)
            image_drawn = draw_pattern(new_pattern_box, image_drawn)
            x,y,w,h = bound_mtp
            image_drawn = cv.putText(image_drawn, 'Box', (x,y), cv.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 3, cv.LINE_AA)

        cv.imshow('dilated', dilated)
        cv.imshow('image_drawn', image_drawn)
        cv.imshow('canny', canny)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cv.destroyAllWindows()