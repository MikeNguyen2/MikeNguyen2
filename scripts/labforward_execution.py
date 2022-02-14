import rospy
rospy.init_node('cam_control')
import pyrealsense2 as rs
import transform_test
import numpy as np
import cv2 as cv
import traceback
import pipette
import random
import time
import bot
import sys
# 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

ep_pipette = pipette.Pipette()
ep_pipette.set_mode(steps=1)

cobotta = bot.MoveGroupInterfaceBot()
cobotta.set_speed(0.7, 0.7)
cobotta.move_e(0.045, 0.2, 0.3, 0, 0, 0)
cobotta.move_h(8.1, 1)

pipeline = rs.pipeline()
config = rs.config()

pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
#device_product_line = str(device.get_info(rs.camera_info.prodbinaruct_line))

config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 6)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 6)

profile = pipeline.start(config)

color_sensor = profile.get_device().query_sensors()[1]
color_sensor.set_option(rs.option.enable_auto_exposure, True)
color_sensor.set_option(rs.option.enable_auto_white_balance, True)

align_to = rs.stream.color
align = rs.align(align_to)

# 0 = corners, 1 = bounding_area, 2 = area, 3 = length, 4 = color_normal, 5 = color_pippett 
# mtp = [[4,6],[40000,140000],[10000,80000],[800,1600],[0,0,0,0,0,0]]
# pippettenhalter_blau = [[4,100],[45000,150000],[5000,95000],[0],[136,61,67,212,199,148]]#[91,71,49,183,129,127]
# colb_green = [[7,100],[20000,60000],[10000,50000],[0],[108,211,123,175,255,255]] 
# colb_red = [[7,100],[35000,85000],[5000,60000],[0],[0 ,37,168,112,148,255]]
# phenol = [[7,100],[10000,30000],[5000,25000],[0],[0 ,37,168,112,148,255]]

# returns a binary image
def get_binar(spectrum,image):
    lower_limit = np.asarray(spectrum[0:3])
    upper_limit = np.asarray(spectrum[3:6])
    return cv.inRange(image.copy(),lower_limit,upper_limit) 

# draws a rotated template on the object
def black_magic(image,margin,margin2):
    global contour
    border_margin = margin
    border_margin2 = margin2
    font = cv.FONT_HERSHEY_SIMPLEX

    img_gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    height, width = image.shape[:2]
    _,contours, hierarchy = cv.findContours(img_gray, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    x, y, w, h = cv.boundingRect(cnt)

    image = cv.rectangle(image, (x, y), (x+w, y+h), (255, 128, 0), 2)

    minAreaRect = cv.minAreaRect(cnt)
    angle = minAreaRect[2]
    points = np.int0(cv.boxPoints(minAreaRect))
    cv.drawContours(image, [points], 0, (0, 128, 255), 2)

    for i in range(len(points)):
        contour = cv.circle(contour, (points[i][0], points[i][1]), 16, (0, 255, 0), 2)
        image = cv.circle(image, (points[i][0], points[i][1]), 16, (0, 255, 0), 2)


    shortest_distance_sq = 10**10
    closest_point = None
    random_point = random.choice(points)
    for point in points:
        if point[0] == random_point[0] and point[1] == random_point[1]: continue

        distance_sq = (random_point[0]-point[0])**2 + (random_point[1]-point[1])**2
        if distance_sq < shortest_distance_sq:
            shortest_distance_sq = distance_sq
            closest_point = point


    pair1 = [random_point, closest_point]

    pair2 = []
    for point in points:
        if point[0] == pair1[0][0] and point[1] == pair1[0][1]: continue
        if point[0] == pair1[1][0] and point[1] == pair1[1][1]: continue

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

    contour = cv.circle(contour, (top_left[0], top_left[1]), 24, (255, 0, 255), 2)
    image = cv.circle(image, (top_left[0], top_left[1]), 24, (255, 0, 255), 2)
    contour = cv.putText(contour, str(round(angle*100.0)/100.0)+'deg', (x-120, y-20),font, 1, (255, 128, 0), 2)
    image = cv.putText(image, str(round(angle*100.0)/100.0)+'deg', (x-120, y-20),font, 1, (255, 128, 0), 2)

    shortest_distance_sq = 10**10
    closest = None
    for point in points:
        if point[0] == top_left[0] and point[1] == top_left[1]: continue

        distance_sq = (top_left[0]-point[0])**2 + (top_left[1]-point[1])**2
        if distance_sq < shortest_distance_sq:
            shortest_distance_sq = distance_sq
            closest = point

    longest_distance_sq = 0
    furthest = None
    for point in points:
        if point[0] == top_left[0] and point[1] == top_left[1]: continue

        distance_sq = (top_left[0]-point[0])**2 + (top_left[1]-point[1])**2
        if distance_sq > longest_distance_sq:
            longest_distance_sq = distance_sq
            furthest = point

    closest2 = None
    for point in points:
        if point[0] == top_left[0] and point[1] == top_left[1]: continue
        if point[0] == furthest[0] and point[1] == furthest[1]: continue
        if point[0] == closest[0] and point[1] == closest[1]: continue
        closest2 = point

    con_closest = [closest[0]-top_left[0], closest[1]-top_left[1]]
    con_closest2 = [closest2[0]-top_left[0], closest2[1]-top_left[1]]

    contour = cv.line(contour, (top_left[0], top_left[1]),(top_left[0]+con_closest[0], top_left[1]+con_closest[1]), (0, 0, 255), 5)
    image = cv.line(image, (top_left[0], top_left[1]),(top_left[0]+con_closest[0], top_left[1]+con_closest[1]), (0, 0, 255), 5)

    contour = cv.line(contour, (top_left[0], top_left[1]),(top_left[0]+con_closest2[0], top_left[1]+con_closest2[1]), (255, 255, 0), 5)
    image = cv.line(image, (top_left[0], top_left[1]),(top_left[0]+con_closest2[0], top_left[1]+con_closest2[1]), (255, 255, 0), 5)

    def mapping(value, a, b, c, d):
        return c + (d - c) * ((value - a) / float(b - a))

    um = 1-border_margin
    lm = border_margin
    um2 = 1-border_margin2
    lm2 = border_margin2
    well_radius = int((minAreaRect[1][0]+minAreaRect[1][1]) * 0.02)
    
    positions=[]
    for j in range(12):
        positions.append([])
        for i in range(8):
            well_x = top_left[0]
            well_x += mapping(i, 0, 8-1, con_closest[0]*lm2, con_closest[0]*um2)
            well_x += mapping(j, 0, 12-1, con_closest2[0]*lm, con_closest2[0]*um)
            
            
            well_y = top_left[1]
            well_y += mapping(i, 0, 8-1, con_closest[1]*lm2, con_closest[1]*um2)
            well_y += mapping(j, 0, 12-1, con_closest2[1]*lm, con_closest2[1]*um)
            
            positions[j].append([int(well_x),int(well_y)])
            cv.circle(contour, (int(well_x), int(well_y)), well_radius, (255, 64, 64), 2)
            cv.circle(image, (int(well_x), int(well_y)), well_radius, (255, 64, 64), 2)

    for i in range(8):
        text_x = top_left[0]
        text_x += mapping(i, 0, 8-1, con_closest[0]*lm2, con_closest[0]*um)
        text_x += con_closest2[0]*lm/4.0
        
        text_y = top_left[1]
        text_y += mapping(i, 0, 8-1, con_closest[1]*lm2, con_closest[1]*um)
        text_y += con_closest2[1]*lm/4.0
        
        contour = cv.putText(contour, chr(i+65), (int(text_x), int(text_y)),font, 0.75, (0, 0, 0), 2)
        image = cv.putText(image, chr(i+65), (int(text_x), int(text_y)),font, 0.75, (0, 0, 0), 2)

    for j in range(12):
        text_x = top_left[0]
        text_x += mapping(j, 0, 12-1, con_closest2[0]*lm, con_closest2[0]*um)
        text_x += con_closest[0]*lm/4.0
        
        text_y = top_left[1]
        text_y += mapping(j, 0, 12-1, con_closest2[1]*lm, con_closest2[1]*um)
        text_y += con_closest[1]*lm/4.0
        
        contour = cv.putText(contour, str(j+1), (int(text_x), int(text_y)),font, 0.75, (0, 0, 0), 2)
        image = cv.putText(image, str(j+1), (int(text_x), int(text_y)),font, 0.75, (0, 0, 0), 2)

    cv.imwrite('mtp5_detected.png', image)
    cv.imshow('MTP', image)
    return positions,top_left,furthest,closest,closest2

# returns a list of the wells
def classify_mtp(contours,hierachies):
    for i in range(len(contours)):
        area = cv.contourArea(contours[i])
        if 1000 < area :
            # data
            x,y,w,h = cv.boundingRect(contours[i])
            length = cv.arcLength(contours[i],False)
            approx = cv.approxPolyDP(contours[i],0.02*length,False)
            corners = len(approx)
            bounding_area = w*h
            # compares to defined objects
            if 4 <= corners <= 6 and 40000 < bounding_area < 140000 and 10000 < area < 80000 and 800 < length < 1600:
                if hierachies[0][i][3]==-1:
                    a = hierachies[0][i][2]
                    j =0
                    
                    M = cv.moments(contours[i])
                    if M['m00'] != 0.0:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                    else:
                        cx,cy = 0,0
                    rect = cv.minAreaRect(contours[i])
                    x,y,w,h = cv.boundingRect(contours[i])
                    bounding_area = w*h
                    con = contours[i]
                    global contour
                    while a != 's' and j < 100:
                        cv.imshow('adap',adap)
                        j=j+1   
                        area = cv.contourArea(contours[a])
                        if 10000 < area:
                            rect = cv.minAreaRect(contours[a])
                            x,y,w,h = cv.boundingRect(contours[a])
                            length = cv.arcLength(contours[a],False)
                            approx = cv.approxPolyDP(contours[a],0.02*length,False)
                            corners = len(approx)
                            bounding_area = w*h
                            if 4 <= corners <= 6 and 30000 < bounding_area < 145000 and 30000 < area < 85000 and 800 < length < 1600:
                                mean = np.mean(img[y:y+h,x:x+w])
                                #print('mtp',bounding_area,area,length,mean)
                                M = cv.moments(contours[a])
                                if M['m00'] != 0.0:
                                    cx = int(M['m10']/M['m00'])
                                    cy = int(M['m01']/M['m00'])
                                else:
                                    cx,cy = 0,0
                                con = contours[a]

                            contour = cv.drawContours(contour,con,-1,(255,0,0),1)   
                            contour = cv.rectangle(contour,(x,y),(x+w,y+h),(0,255,0),1)
                            contour = cv.putText(contour,'Mtp',(x,y), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,225),2)
                            
                            blank = np.zeros(contour.shape,dtype=np.uint8)
                            blank = cv.drawContours(blank,con,-1,(255,255,255),1)
                            cv.imshow('contours_on_black',blank)
                            cv.imshow('contour',contour)
                            
                            wells,_,_,_,_ = black_magic(blank,0.11,0.14)
                            return wells

                        if hierachies[0][a][0] != -1:
                            a = hierachies[0][a][0]
                        else:
                            a = 's'
                        
                    
                    contour = cv.drawContours(contour,con,-1,(255,0,0),1)   
                    contour = cv.rectangle(contour,(x,y),(x+w,y+h),(0,255,0),1)
                    contour = cv.putText(contour,'Mtp',(x,y), cv.FONT_HERSHEY_SIMPLEX,0.75,(0,0,225),2)
                    cv.imshow('contour',contour)

# returns a list of the holes
def classify_box():
    global contour
    binar = get_binar([91,71,49,183,129,127],img)
    cv.imshow('box', binar)
    _,contours,hierachies = cv.findContours(binar,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        bounding_area = w*h
        if 45000 < bounding_area < 150000 and 5000 < area < 19500 and 0.1*h < w < 1.9*h:
            #print('box',bounding_area,area)
            minAreaRect = cv.minAreaRect(cnt)
            angle = minAreaRect[2]
            points = np.int0(cv.boxPoints(minAreaRect))
            #print(points)
            min_x = img.shape[1]
            min_y = img.shape[0]
            max_x = 0
            max_y = 0
            for point in points:
                if point[0] < min_x:
                    min_x = point[0]
                if point[1] < min_y:
                    min_y = point[1]
                if point[0] > max_x:
                    max_x = point[0]
                if point[1] > max_y:
                    max_y = point[1]    
            contour = cv.rectangle(contour,(min_x,min_y),(max_x,max_y),(0,255,0),1)

            contour = cv.drawContours(contour,cnt,-1,(255,0,0),1)
            for point in points:
                contour = cv.circle(contour,(point[0],point[1]),10,(255,255,0),1)
            blank = np.zeros(contour.shape,dtype=np.uint8)
            blank = cv.line(blank, (points[0][0],points[0][1]),(points[1][0],points[1][1]), (255, 255, 255), 5)
            blank = cv.line(blank, (points[1][0],points[1][1]),(points[2][0],points[2][1]), (255, 255, 255), 5)
            blank = cv.line(blank, (points[2][0],points[2][1]),(points[3][0],points[3][1]), (255, 255, 255), 5)
            blank = cv.line(blank, (points[3][0],points[3][1]),(points[0][0],points[0][1]), (255, 255, 255), 5)
            cv.imshow('blank',blank)
            tips,start,end,start2,end2 = black_magic(blank,0.08,0.12)

            global img_boxless
            img_boxless = cv.circle(img_boxless, (start[0], start[1]), 16, (0, 255, 0), 4)
            img_boxless = cv.circle(img_boxless, (start2[0], start2[1]), 16, (0, 255, 0), 4)
            img_boxless = cv.circle(img_boxless, (end[0], end[1]), 16, (0, 255, 0), 4)
            img_boxless = cv.circle(img_boxless, (end2[0], end2[1]), 16, (0, 255, 0), 4)
            img_boxless = cv.circle(img_boxless, ((end2[0]+end[0])/2, (end2[1]+end[1])/2), 16, (0, 255, 0), 4)
            img_boxless = cv.circle(img_boxless, ((start2[0]+start[0])/2,(start2[1]+start[1])/2), 16, (0, 255, 0), 4)
            cv.imshow('boxless',img_boxless)
            return tips

# returns the center of the colb
def look_for_entry():
    global contour
    binar = get_binar([108,211,123,175,255,255],img)# gruen [41,195,124,138,255,227],![108,211,123,175,255,255]
    cv.imshow('green_colb',binar)
    _,contours,_ = cv.findContours(binar,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        bounding_area = w*h
        if 10000 < area < 50000 and 20000 < bounding_area < 60000:
            # print('colb',bounding_area, area)
            x,y,w,h = cv.boundingRect(cnt)
            M = cv.moments(cnt)
            if M['m00'] != 0.0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx,cy = 0,0

            contour = cv.putText(contour,'Top Colb', (x, y),cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            contour = cv.drawContours(contour,cnt,-1,(255,0,0),1)
            contour = cv.rectangle(contour,(x,y),(x+w,y+h),(0,255,0),1)
            return [cx,cy]

def look_for_entry2():
    global contour
    binar = get_binar([0 ,37,168,112,148,255],img)
    cv.imshow('red_colb',binar)
    _,contours,_ = cv.findContours(binar,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        bounding_area = w*h
        if 35000 < bounding_area < 85000 and 5000 < area < 60000 :
            x,y,w,h = cv.boundingRect(cnt)
            bounding_area = w*h
            #print( bounding_area,area)
            M = cv.moments(cnt)
            if M['m00'] != 0.0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx,cy = 0,0

            contour = cv.putText(contour,'Top Colb', (x, y),cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            contour = cv.drawContours(contour,cnt,-1,(255,0,0),1)
            contour = cv.rectangle(contour,(x,y),(x+w,y+h),(0,255,0),1)
            return [cx,cy]

# returns the center of the phenol
def look_for_bottom():
    global contour
    binar = get_binar([0 ,37,168,112,148,255],img)
    cv.imshow('red_bottom',binar)
    _,contours,_ = cv.findContours(binar,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv.contourArea(cnt)
        x,y,w,h = cv.boundingRect(cnt)
        bounding_area = w*h
        if 10000 < bounding_area < 30000 and 5000 < area < 25000 :
            #print('phenol',bounding_area, area)
            x,y,w,h = cv.boundingRect(cnt)
            M = cv.moments(cnt)
            if M['m00'] != 0.0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
            else:
                cx,cy = 0,0

            contour = cv.putText(contour,'Bottom Phenol', (x, y),cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            contour = cv.drawContours(contour,cnt,-1,(255,0,0),1)
            contour = cv.rectangle(contour,(x,y),(x+w,y+h),(0,255,0),1)
            return [x+w/2,y+h/2]


# # searches the phenol
# pos_phenol = []
# counter = 0
# try:
#     while counter < 10:
#         frames = pipeline.wait_for_frames()
#         frames = align.process(frames)
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         depth_image = np.asanyarray(depth_frame.get_data())
#         img = np.asanyarray(color_frame.get_data())
#         img_boxless = img.copy()
#         contour = img.copy()
    
#         center_phenol = look_for_bottom()
#         if center_phenol:
#             pos_phenol.append(center_phenol)
#             counter = counter + 1
# except:
#     traceback.print_exc()

# average_phenol = [0,0]

# for i in range(10):
#     for j in range(2):
#         average_phenol[j] += pos_phenol[i][j]

# for j in range(2):
#     average_phenol[j] = average_phenol[j]/10

# # fill the pipette with phenol
# try:
#     point = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, average_phenol, 0.401)
#     cam_x, cam_y, cam_z = transform_test.get_camera_position()
#     x = cam_x + point[0] # x = point[1]
#     y = cam_y - point[1] # y = point[0]+0.35
#     z = cam_z - point[2] # z = 0.5-point[2]
#     print('camera', cam_x, cam_y, cam_z)
#     print('point', point[0], point[1], point[2])
#     print('cobotta target:', x, y, z)
#     #cobotta.move_e(x, y, z, 0, 0, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.15, -90, -90, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.12, -90, -90, 0)
#     for stepping in range(3):
#         ep_pipette.step()
#         time.sleep(2)
#     cobotta.move_e(x-0.019, y-0.074, 0.15, -90, -90, 0)
#     cobotta.move_e(0.045, 0.25, 0.3, 0, 0, 0)
#     sys.exit()
# except:
#     traceback.print_exc()
#     #sys.exit()

# # searches the mtp
# pos_mtp =[]
# counter = 0
# try:
#     while counter < 10:
#         frames = pipeline.wait_for_frames()
#         frames = align.process(frames)
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         depth_image = np.asanyarray(depth_frame.get_data())
#         img = np.asanyarray(color_frame.get_data())
#         img_boxless = img.copy()
#         contour = img.copy()
    
#         #center_green_colb,center_red_colb, center_phenol,tips = look_for_entry(),look_for_entry2(), look_for_bottom(),classify_box()

#         gray = cv.cvtColor(img_boxless.copy(),cv.COLOR_BGR2GRAY)
#         adap = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
#         _,adap = cv.threshold(adap, 175, 255, cv.THRESH_BINARY_INV) #comment if white background

#         _,contours,hierachies = cv.findContours(adap,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
#         wells = classify_mtp(contours,hierachies)
  
#         bound = contour

#         cv.imshow('original',img)
#         cv.imshow('adap',adap)
#         cv.namedWindow('Buddy View', cv.WINDOW_AUTOSIZE)
#         cv.imshow('Buddy View', bound)

#         if wells:
#             pos_mtp.append(wells)
#             counter = counter + 1

#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
# except:
#     traceback.print_exc()

# # calculates the average
# average_pos = []
# for i in range(12):
#     average_pos.append([])
#     for j in range(8):
#         average_pos[i].append([0,0])
#         for k in range(2):
#             for l in range(10):
#                 average_pos[i][j][k] += pos_mtp[l][i][j][k]

# for i in range(12):
#     for j in range(8):
#         for k in range(2):
#             average_pos[i][j][k]=average_pos[i][j][k]/10

# # pippetting
# try:
#     point = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, average_pos[0][0], 0.401)
#     cam_x, cam_y, cam_z = transform_test.get_camera_position()
#     x = cam_x + point[0] # x = point[1]
#     y = cam_y - point[1] # y = point[0]+0.35
#     z = cam_z - point[2] # z = 0.5-point[2]
#     print('camera', cam_x, cam_y, cam_z)
#     print('point', point[0], point[1], point[2])
#     print('cobotta target:', x, y, z)
#     #cobotta.move_e(x, y, z, 0, 0, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.15, -90, -90, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.075, -90, -90, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.065, -90, -90, 0)
#     ep_pipette.step()
#     time.sleep(1)
#     cobotta.move_e(x-0.019, y-0.074, 0.075, -90, -90, 0)
#     cobotta.move_e(x-0.019, y-0.074, 0.15, -90, -90, 0)
#     sys.exit()
# except:
#     traceback.print_exc()
#     #sys.exit()

# finally:
#     pipeline.stop()

cam_x, cam_y, cam_z = transform_test.get_camera_position()

def get_position(position, z_depth):
    point = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics, position, z_depth)
    x = cam_x + point[0] # x = point[1]
    y = cam_y - point[1] # y = point[0]+0.35
    z = cam_z - point[2] # z = 0.5-point[2]

    return x,y,z

pos_tips = []
pos_wells = []
pos_phenol = []
pos_colb = []
points_mtp = []

def look_for(obj):
    counter = 0
    while counter < 10:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)
        global depth_frame
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        global img
        global contour
        global img_boxless
        img = np.asanyarray(color_frame.get_data())
        img_boxless = img.copy()
        contour = img.copy()

        if obj == 'tips':
            tips = classify_box()
            if tips:
                pos_tips.append(tips)
                counter = counter + 1

        if obj == 'wells':
            tips = classify_box()
            gray = cv.cvtColor(img_boxless.copy(),cv.COLOR_BGR2GRAY)
            global adap
            adap = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,3)
            _,adap = cv.threshold(adap, 175, 255, cv.THRESH_BINARY_INV) #comment if white background

            _,contours,hierachies = cv.findContours(adap,cv.RETR_TREE,cv.CHAIN_APPROX_NONE)
            #wells,top_left,furthest,closest,closest2  = classify_mtp(contours,hierachies)
            wells = classify_mtp(contours,hierachies)
            if wells:
                pos_wells.append(wells)
                counter = counter + 1
                #points_mtp.append([top_left,furthest,closest,closest2])

        if obj == 'phenol':
            center_phenol = look_for_bottom()
            if center_phenol:
                pos_phenol.append(center_phenol)
                counter = counter + 1

        if obj == 'colb':
            center_colb = look_for_entry2()
            if center_colb:
                pos_colb.append(center_colb)
                counter = counter + 1

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def calc_average_96(list):
    average_pos = []
    for i in range(12):
        average_pos.append([])
        for j in range(8):
            average_pos[i].append([0,0])
            for k in range(2):
                for l in range(10):
                    average_pos[i][j][k] += list[l][i][j][k]
    for i in range(12):
        for j in range(8):
            for k in range(2):
                average_pos[i][j][k] /= 10 
    return average_pos

def calc_average_1(list):
    average_pos = [0,0]
    for k in range(10):
        for i in range(2):
            average_pos[i] += list[k][i]
    for j in range(2):
        average_pos[j] /= 10
    return average_pos

def calc_average_points(list):
    average_pos = [[0,0],[0,0],[0,0],[0,0]]
    for l in range(10):
        for i in range(4):
            for j in range(2):
                average_pos[i][j] += points_mtp[l][i][j]
    for i in range(4):
            for j in range(2):
                average_pos[i][j] /= 10
    return average_pos

look_for('tips')
look_for('wells')
look_for('phenol')
look_for('colb')

average_tips = calc_average_96(pos_tips)
average_wells = calc_average_96(pos_wells)
average_phenol = calc_average_1(pos_phenol)
average_colb = calc_average_1(pos_colb)

def move_to(x,y,z):
    try:
        cobotta.move_e(x-0.019, y-0.074, 0.2, -90, -90, 0)
    except:
        traceback.print_exc()
        sys.exit()

def leave_tip():
    # action
    pass

def get_tip(a,b):
    x,y,z = get_position(average_tips[a][b], 0.35)
    return x, y, z
    # move_to(x,y,z)
    # action

def leave_tip(a,b):
    x,y,z = get_position(average_tips[a][b], 0.35)
    return x, y, z
    # move_to(x,y,z)
    # action

def get_water():
    x,y,z = get_position(average_colb, 0.25)
    return x, y, z
    # move_to(x,y,z)
    # action

def get_phenol():
    x,y,z = get_position(average_phenol, 0.351)
    return x, y, z
    # move_to(x,y,z)
    # action

def pipette_water(a,b):
    x,y,z = get_position(average_wells[a][b], 0.401)
    return x, y, z
    # move_to(x,y,z) 
    # for i in range(4):
    #     x,y,z = get_position(average_wells[a][b+i])
    #     #cobotta.move_e(x-0.019, y-0.074, 0.075, -90, -90, 0) 
    #     # action 0.5 wird erhoeht
    #     #cobotta.move_e(x-0.019, y-0.074, 0.01, -90, -90, 0)
    #     print(cobotta.group.get_current_pose().pose)
    #     cobotta.move_l([
    #         (x-0.019, y-0.074, 0.075, -90, -90, 0),
    #         (x-0.019, y-0.074, 0.05, -90, -90, 0)
    #     ])
    #     #ep_pipette.step()

def pipette_phenol(a,b):
    x,y,z = get_position(average_wells[a][b], 0.401)
    return x, y, z
    # move_to(x,y,z)
    # for i in range(4):
    #     x,y,z = get_position(average_wells[0][i])
    #     cobotta.move_e(x-0.019, y-0.074, 0.075, -90, -90, 0) 
    #     # action 0.5 wird verringert
    #     cobotta.move_e(x-0.019, y-0.074, 0.01, -90, -90, 0)


print(average_colb)
#average_points = calc_average_points(points_mtp)
# # for letters in range(12):
# #     for half in range(0,5,4):
print(get_position(pos_phenol[0],0.4))
print(get_position(pos_wells[0][0][0],0.4))
print(get_position(average_tips[0][0], 0.35))
print(get_position(average_wells[0][0], 0.401))
print(get_position(average_phenol, 0.401))
print(get_position(average_colb, 0.25))

tip_a1 = get_tip(0, 0)
water = get_water()
mtp_a1 = pipette_water(0, 0)
tip_b1 = get_tip(0, 1)
phenol = get_phenol()

def gen(pos):
    return (pos[0]-0.019, pos[1]-0.074, pos[2])

tip_a1 = gen(tip_a1)
water = gen(water)
mtp_a1 = gen(mtp_a1)
tip_b1 = gen(tip_b1)
phenol = gen(phenol)

cobotta.move_l([
    (tip_a1[0], tip_a1[1], 0.25, -90, -90, 0),
    (tip_a1[0], tip_a1[1], 0.20, -90, -90, 0),
    (tip_a1[0], tip_a1[1], 0.25, -90, -90, 0),
])

cobotta.move_l([
    (water[0], water[1], 0.25, -90, -90, 0),
    (water[0], water[1], 0.20, -90, -90, 0),
    (water[0], water[1], 0.25, -90, -90, 0),
])

cobotta.move_l([
    (mtp_a1[0], mtp_a1[1], 0.25, -90, -90, 0),
    (mtp_a1[0], mtp_a1[1], 0.20, -90, -90, 0),
    (mtp_a1[0], mtp_a1[1], 0.25, -90, -90, 0),

    (tip_a1[0], tip_a1[1], 0.25, -90, -90, 0),
])

cobotta.move_l([
    (tip_b1[0], tip_b1[1], 0.25, -90, -90, 0),
    (tip_b1[0], tip_b1[1], 0.20, -90, -90, 0),
    (tip_b1[0], tip_b1[1], 0.25, -90, -90, 0)
])

cobotta.move_l([
    (phenol[0], phenol[1], 0.25, -90, -90, 0),
    (phenol[0], phenol[1], 0.20, -90, -90, 0),
    (phenol[0], phenol[1], 0.25, -90, -90, 0),
])

cobotta.move_l([
    (mtp_a1[0], mtp_a1[1], 0.25, -90, -90, 0),
    (mtp_a1[0], mtp_a1[1], 0.20, -90, -90, 0),
    (mtp_a1[0], mtp_a1[1], 0.25, -90, -90, 0),

    (tip_b1[0], tip_b1[1], 0.25, -90, -90, 0)
])

'''get_tip(0,0)
time.sleep(2)
get_water()
time.sleep(2)
pipette_water(0,0)
time.sleep(2)
leave_tip(0,0)
time.sleep(2)
get_tip(0,1)
time.sleep(2)
get_phenol()
time.sleep(2)
pipette_phenol(0,0)
time.sleep(2)
leave_tip(0,1)'''

# look_for('mtp')
# look_for('reader')
# get_mtp
# input_mtp

# for fill_counter < 24
    # look_pip -> calc_average -> move_to -> take action 
    # look_water -> calc_average -> move_to -> take action 
    # look_well -> calc_average -> move_to -> take action(A1,A2,A3,A4)
    # look_pip -> calc_average -> move_to -> take action 
    # look_well -> calc_average -> move_to -> take action 
    # look_mtp -> calc_average -> move_to -> take action
# look_mtp -> get_up 
# look_reader -> put_in