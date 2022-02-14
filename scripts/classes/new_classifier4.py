import camera
import painter
import cv2 as cv
import numpy as np

# handle auf dem PC nicht erkannt

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

def check_negative(circles):
    for i in range(len(circles)):
        for j in range(len(circles[0])):
            if circles[i][j][1] < 0:
                print(i,j, 'is under 0')

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

    width1 = float(max(np.absolute(top_right[0]-top_left[0]),1))
    height1 = float(max(np.absolute(top_right[1]-top_left[1]),1))
    width2 = float(max(np.absolute(bottom_left[0]-top_left[0]),1))
    height2 = float(max(np.absolute(bottom_left[1]-top_left[1]),1))

    top_form = np.power(margin1,2)*(np.power(width1,2)+np.power(height1,2))
    bottom_form = 1 + np.power(height1/width1,2)
    margin_x = np.absolute(np.sqrt(top_form/bottom_form))
    margin_y = margin_x * height1/width1

    top_form2 = np.power(margin2,2)*(np.power(width2,2)+np.power(height2,2))
    bottom_form2 = 1 + np.power(height2/width2,2)
    margin_x2 = np.absolute(np.sqrt(top_form2/bottom_form2))
    margin_y2 = margin_x2 * height2/width2

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

def is_rectangle(width, height, w_min,w_max, h_min, h_max):
    if w_max < max(width,height) or max(width,height) < w_min: return False
    if h_max < min(width,height) or min(width,height) < h_min: return False
    return True

def has_corners(contour, conrer_min, corner_max):
    length = cv.arcLength(contour,False)
    approx = cv.approxPolyDP(contour,0.02*length,False)
    corners = len(approx)
    if conrer_min <= corners <= corner_max:
        return True

def is_centered(center_x, center_y):
    offset = 1000
    img_cen_x = 640
    img_cen_y = 360
    if img_cen_x-offset > center_x or center_x > img_cen_x+offset: return False
    if img_cen_y-offset > center_y or center_y > img_cen_y+offset: return False
    return True

def has_circles(img, dist ,canny_limit, lower_limit, min_radius, max_radius, x, y):
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)
    dilated = cv.dilate(gblur, (5,5))

    circles_zoom = cv.HoughCircles(dilated, cv.HOUGH_GRADIENT, 1, dist,
                                param1=canny_limit, param2=lower_limit,
                                minRadius=min_radius, maxRadius=max_radius)
    circles_seen = []
    if circles_zoom is not None:
        for circle_zoom in circles_zoom[0]:
            circle_new = (int(circle_zoom[0]+x), int(circle_zoom[1]+y), int(circle_zoom[2]))
            circles_seen.append(circle_new)
    
    if len(circles_seen) < 20: return False
    return circles_seen

def compare_pattern(pattern, circles_seen):
    new_pattern = pattern[:]
    for row in range(len(pattern)):
        for column in range(len(pattern[0])):
            circle_calc = pattern[row][column]
            for circle_seen in circles_seen:
                distance = np.sqrt(np.power(circle_calc[0]-circle_seen[0],2)+np.power(circle_calc[1]-circle_seen[1],2))
                if distance > 12: continue
                circle_x = circle_seen[0]
                circle_y = circle_seen[1]
                radius = circle_seen[2]
                color = (255, 0, 255)
                new_pattern[row][column] = [circle_x, circle_y, radius, color]
                break
    return new_pattern

def get_avg_depth(pattern, depth_frame):
    depth_sum = 0
    depth_div = 0
    for row in range(len(pattern)):
        for column in range(len(pattern[0])):
            x,y = pattern[row][column][:2]
            if x < 0 or y < 0: continue
            if depth_frame.get_distance(x,y) == 0: continue
            # if camera.get_depth(y,x,depth_frame) == 0: continue
            depth_sum += depth_frame.get_distance(x,y) 
            depth_div += 1
    try:
        avg_depth = depth_sum/depth_div
        return avg_depth
    except:
        # print("0 division")
        return 0

def classify_mtp(contours, color_image, depth_frame):
    # rectangle
    ofs = 10
    w_min = 270 - ofs
    w_max = 290 + ofs
    h_min = 180 - ofs
    h_max = 200 + ofs
    z_min = 0.38
    z_max = 0.4

    # wells
    dist = 10
    canny_limit = 100
    lower_limit = 10
    min_radius = 8
    max_radius = 12

    for i in range(len(contours)):
        if has_corners(contours[i], 4, 6) is False: continue
        x,y,w,h = cv.boundingRect(contours[i])
        minrect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        bottom, left, top, right = box

        center_y = (2*y+h)/2
        center_x = (2*x+w)/2

        width = np.sqrt(np.power(bottom[0]-left[0],2)+np.power(bottom[1]-left[1],2))
        height = np.sqrt(np.power(left[0]-top[0],2)+np.power(left[1]-top[1],2))

        img_zoom = color_image[y:y+h,x:x+w]

        if is_rectangle(width, height, w_min,w_max, h_min,h_max) is False: continue
        if is_centered(center_x, center_y) is False: continue

        circles_seen = has_circles(img_zoom, dist, canny_limit, lower_limit, min_radius, max_radius, x, y)
        if circles_seen is False: continue

        pattern = simple_pattern(box, 8, 12, 0.11, 0.13)
        new_pattern = compare_pattern(pattern, circles_seen)

        avg_depth = get_avg_depth(new_pattern, depth_frame)
        if z_min < avg_depth < z_max:
            print('mtp: ', width, height, avg_depth)
            return (box, contours[i], new_pattern)
        
def classify_box_blue(contours, color_image, depth_frame):    
    # rectangle
    ofs = 30
    w_min = 310 - ofs
    w_max = 330 + ofs
    h_min = 210 - ofs
    h_max = 230 + ofs
    z_min = 0.32
    z_max = 0.36

    # wells
    dist = 15
    canny_limit = 75
    lower_limit = 5
    min_radius = 10
    max_radius = 16

    for i in range(len(contours)):
        if has_corners(contours[i], 4, 4) is False: continue
        x,y,w,h = cv.boundingRect(contours[i])
        minrect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        bottom, left, top, right = box

        center_y = (2*y+h)/2
        center_x = (2*x+w)/2

        width = np.sqrt(np.power(bottom[0]-left[0],2)+np.power(bottom[1]-left[1],2))
        height = np.sqrt(np.power(left[0]-top[0],2)+np.power(left[1]-top[1],2))

        img_zoom = color_image[y:y+h,x:x+w]

        if is_rectangle(width, height, w_min,w_max, h_min,h_max) is False: continue
        if is_centered(center_x, center_y) is False: continue

        circles_seen = has_circles(img_zoom, dist, canny_limit, lower_limit, min_radius, max_radius, x, y)
        if circles_seen is False: continue
        
        pattern = simple_pattern(box, 8, 12, 0.08, 0.12)
        new_pattern = compare_pattern(pattern, circles_seen)
        avg_depth = get_avg_depth(new_pattern, depth_frame)
        
        if z_min < avg_depth < z_max:
            print('blue: ', width, height, avg_depth)
            return (box, contours[i], new_pattern)

def classify_box_yellow(contours, color_image, depth_frame):    
    # rectangle
    ofs = 5
    w_min = 290 - ofs
    w_max = 310 + ofs
    h_min = 190 - ofs
    h_max = 210 + ofs
    z_min = 0.35
    z_max = 0.37  

    # wells
    dist = 10
    canny_limit = 50
    lower_limit = 10
    min_radius = 5
    max_radius = 10

    for i in range(len(contours)):
        if has_corners(contours[i], 4, 4) is False: continue
        x,y,w,h = cv.boundingRect(contours[i])
        minrect = cv.minAreaRect(contours[i])
        box = cv.boxPoints(minrect)
        box = np.int0(box)
        bottom, left, top, right = box

        center_y = (2*y+h)/2
        center_x = (2*x+w)/2

        width = np.sqrt(np.power(bottom[0]-left[0],2)+np.power(bottom[1]-left[1],2))
        height = np.sqrt(np.power(left[0]-top[0],2)+np.power(left[1]-top[1],2))

        img_zoom = color_image[y:y+h,x:x+w]

        if is_rectangle(width, height, w_min,w_max, h_min,h_max) is False: continue
        if is_centered(center_x, center_y) is False: continue

        circles_seen = has_circles(img_zoom, dist, canny_limit, lower_limit, min_radius, max_radius, x, y)
        if circles_seen is False: continue
        
        pattern = simple_pattern(box, 8, 12, 0.08, 0.12)
        new_pattern = compare_pattern(pattern, circles_seen)
        avg_depth = get_avg_depth(new_pattern, depth_frame)
        
        if z_min < avg_depth < z_max:
            print('yellow: ', width, height, avg_depth)
            return (box, contours[i], new_pattern)

def classify_phenol(dilated):
    dist = 20
    canny_limit = 100
    lower_limit = 25
    min_radius = 60
    max_radius = 70

    dist_small = 20
    canny_limit_small = 100
    lower_limit_small = 20
    min_radius_small = 40
    max_radius_small = 60
    circles = cv.HoughCircles(dilated, cv.HOUGH_GRADIENT, 1, dist,
                                    param1=canny_limit, param2=lower_limit,
                                    minRadius=min_radius, maxRadius=max_radius)

    if circles is not None:
        circles = np.uint16(np.around(circles[0]))
        for circle in circles:
            center = (circle[0], circle[1])
            radius = circle[2]
            x,x2 = circle[0] - radius, circle[0] + radius
            y,y2 = circle[1] - radius, circle[1] + radius
            dilated_zoom = dilated[y:y2,x:x2]
            circles_small = cv.HoughCircles(dilated_zoom, cv.HOUGH_GRADIENT, 1, dist_small,
                                    param1=canny_limit_small, param2=lower_limit_small,
                                    minRadius=min_radius_small, maxRadius=max_radius_small)
            if circles_small is not None:
                if circles_small[0] is not None:
                    return (circle[0],circle[1],circle[2],(0,0,255))

def draw_rec_with_circ(image, contour, circles, name):
    # image = cv.drawContours(image, contour, -1, (255,255,255), 3)
    minrect = cv.minAreaRect(contour)
    box = cv.boxPoints(minrect)
    box = np.int0(box)
    x,y,w,h = cv.boundingRect(box)

    image = cv.putText(image, name, (x,y), cv.FONT_HERSHEY_SIMPLEX,1, (255,255,0), 3, cv.LINE_AA)
    image = cv.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 3)

    for i in range(4):
        cv.line(image_drawn, (box[i][0], box[i][1]), (box[i-1][0], box[i-1][1]), (255, 255, 0), 2, cv.LINE_AA)
    
    for row in range(len(circles)):
        for column in range(len(circles[0])):
            circle = circles[row][column]
            cv.circle(image, (circle[0], circle[1]), circle[2], circle[3], 2)
            image = cv.putText(image, str(int(circle[4])), (circle[0],circle[1]), cv.FONT_HERSHEY_SIMPLEX, 0.5,(255,255,0), 1, cv.LINE_AA)
    return image

def check_tips_depth(circles, depth_frame, depth_min, depth_max):
    new_circles = []
    for row in range(len(circles)):
        new_circles.append([])
        for column in range(len(circles[0])):
            depth = 0.0
            circle = circles[row][column]
            x,y = circle[:2]
            for i in range(-2,3):
                for j in range(-2,3):
                    depth += depth_frame.get_distance(x+j,y+i)
            depth /= (0.01*25)
            new_circle = circle[:]
            new_circle.append(depth)
            if depth_max > depth > depth_min:
                new_circle[3] = (255,0,0)
            new_circles[row].append(new_circle)
    return new_circles

def check_tips_color(circles, img, blue_min, green_min, red_min):
    new_circles = []
    for row in range(len(circles)):
        new_circles.append([])
        for column in range(len(circles[0])):
            circle = circles[row][column]
            x,y = circle[:2]
            blue = 0 
            green = 0
            red = 0
            for i in range(-2,3):
                for j in range(-2,3):
                    (b,g,r) = img[y][x]
                    blue += b
                    green += g
                    red += r
            blue /= 25
            green /= 25
            red /= 25
            new_circle = circle[:]
            new_circle.append(999)
            if blue >= blue_min and green >= green_min and red >= red_min:
                new_circle[3] = (255,0,0)
            new_circles[row].append(new_circle)
    return new_circles

if __name__ == "__main__":
    my_camera = camera.Camera()
    while True:
        (ORIGINAL, DEPTH_FRAME) = my_camera.get_images()
        # ORIGINAL = cv.imread('/home/mike/training/batch_2/training_image65.jpeg')
        # scaledarray = (ORIGINAL_DEPTH_IMAGE/np.max(ORIGINAL_DEPTH_IMAGE))*255

        image_drawn = ORIGINAL.copy()
        gray = cv.cvtColor(ORIGINAL.copy(), cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray,(3,3), cv.BORDER_DEFAULT)

        lower, upper = auto_limit(gblur, 0.33)
        canny = cv.Canny(gblur, lower, upper)
        kernel_big = np.ones((11,11), np.uint8)
        kernel = np.ones((11,11), np.uint8)
        kernel_small = np.ones((3,3), np.uint8)
        #kernel = (37,37)
        dilated = cv.dilate(canny, kernel_big)
        closing = cv.erode(dilated,kernel)
        #closing = cv.erode(closing,kernel_small)
        opening = cv.erode(closing,kernel)
        opening = cv.dilate(opening,kernel)
        cv.imshow('closing', closing)
        cv.imshow('opening', opening)
        _, contours, hierachies = cv.findContours(canny, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)
        
        mtp = classify_mtp(contours, ORIGINAL, DEPTH_FRAME)
        box_blue = classify_box_blue(contours, ORIGINAL, DEPTH_FRAME)
        box_yellow = classify_box_yellow(contours, ORIGINAL, DEPTH_FRAME)
        # phenol = classify_phenol(dilated)

        if mtp is not None:
            (box_points, contour, pattern) = mtp
            x,y,w,h = cv.boundingRect(box_points)
            new_pattern = check_tips_depth(pattern, DEPTH_FRAME, 37, 40)
            image_drawn = draw_rec_with_circ(image_drawn, contour, new_pattern, 'Mtp')
            
        if box_blue is not None:
            (box_points, contour, pattern) = box_blue
            x, y, w, h = cv.boundingRect(box_points)
            new_pattern = check_tips_depth(pattern, DEPTH_FRAME, 32, 35)
            # new_pattern = check_tips_color(pattern, ORIGINAL, 100, 100, 100)
            image_drawn = draw_rec_with_circ(image_drawn, contour, new_pattern, 'Box_Blue')  

        if box_yellow is not None:
            (box_points, contour, pattern) = box_yellow
            x, y, w, h = cv.boundingRect(box_points)
            # new_pattern = check_tips_depth(pattern, DEPTH_FRAME, 35, 37)
            new_pattern = check_tips_color(pattern, ORIGINAL, 0, 145, 195)
            image_drawn = draw_rec_with_circ(image_drawn, contour, new_pattern, 'Box_Yellow')  

        # if phenol is not None:
        #     (cx,cy,r,color) = phenol
        #     cv.circle(image_drawn,(cx,cy),r,color,2)

        #cv.imshow('ORIGINAL', ORIGINAL)
        cv.imshow('dilated', dilated)
        #cv.imshow('canny', canny)
        cv.imshow('image_drawn', image_drawn)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break