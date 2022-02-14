def east():
    # import the necessary packages
    from imutils.object_detection import non_max_suppression
    import numpy as np
    import argparse
    import time
    import cv2

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", type=str,
    	help="path to input image")
    ap.add_argument("-east", "--east", type=str,
    	help="path to input EAST text detector")
    ap.add_argument("-c", "--min-confidence", type=float, default=0.5,
    	help="minimum probability required to inspect a region")
    ap.add_argument("-w", "--width", type=int, default=320,
    	help="resized image width (should be multiple of 32)")
    ap.add_argument("-e", "--height", type=int, default=320,
    	help="resized image height (should be multiple of 32)")
    args = vars(ap.parse_args())

    # load the input image and grab the image dimensions
    image = cv2.imread('/home/mike/cobot/scripts/classes/MTP/lab.png')
    orig = image.copy()
    (H, W) = image.shape[:2]

    # set the new width and height and then determine the ratio in change
    # for both the width and height
    (newW, newH) = (args["width"], args["height"])
    rW = W / float(newW)
    rH = H / float(newH)

    # resize the image and grab the new image dimensions
    image = cv2.resize(image, (newW, newH))
    (H, W) = image.shape[:2]

    # define the two output layer names for the EAST detector model that
    # we are interested -- the first is the output probabilities and the
    # second can be used to derive the bounding box coordinates of text
    layerNames = [
    	"feature_fusion/Conv_7/Sigmoid",
    	"feature_fusion/concat_3"]

    # load the pre-trained EAST text detector
    print("[INFO] loading EAST text detector...")
    net = cv2.dnn.readNet(args["east"])

    # construct a blob from the image and then perform a forward pass of
    # the model to obtain the two output layer sets
    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    start = time.time()
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)
    end = time.time()

    # show timing information on text prediction
    print("[INFO] text detection took {:.6f} seconds".format(end - start))

    # grab the number of rows and columns from the scores volume, then
    # initialize our set of bounding box rectangles and corresponding
    # confidence scores
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # loop over the number of rows
    for y in range(0, numRows):
        # extract the scores (probabilities), followed by the geometrical
        # data used to derive potential bounding box coordinates that
        # surround text
        scoresData = scores[0, 0, y]
        xData0 = geometry[0, 0, y]
        xData1 = geometry[0, 1, y]
        xData2 = geometry[0, 2, y]
        xData3 = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

    	# loop over the number of columns
        for x in range(0, numCols):
            # if our score does not have sufficient probability, ignore it
            if scoresData[x] < args["min_confidence"]:
                continue
            # compute the offset factor as our resulting feature maps will
            # be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)
            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)
            # use the geometry volume to derive the width and height of
            # the bounding box
            h = xData0[x] + xData2[x]
            w = xData1[x] + xData3[x]
            # compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
            endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
            startX = int(endX - w)
            startY = int(endY - h)
            # add the bounding box coordinates and probability score to
            # our respective lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    # loop over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
    	# scale the bounding box coordinates based on the respective
    	# ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)
        # draw the bounding box on the image
        cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # show the output image
    cv2.imshow("Text Detection", orig)
    cv2.waitKey(0)

import cv2 as cv
import camera
import time
import numpy as np
import camera

def easy():
    import easyocr
    reader = easyocr.Reader(['en'])
    while True:
        img = my_camera.get_image()
            
        result = reader.readtext(img)
        print(result)

        for text in result:
            rec,txt,prob = text
            tl,_,br,_ = rec
            cv.rectangle(img,(tl),(br),(255,255,0),3)
        cv.imshow('result',img)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

def tesseract():
    import pytesseract
    custom_config = r'--oem 3 --psm 6'
    # print(pytesseract.get_languages(config=custom_config))
    # print(pytesseract.image_to_string('/home/mike/cobot/scripts/classes/MTP/text.jpg', timeout=2))
    # print(pytesseract.image_to_boxes('/home/mike/cobot/scripts/classes/MTP/text.jpg'))
    # cv.imshow('a', cv.imread('/home/mike/cobot/scripts/classes/MTP/text.jpg'))

    while True:
        img = my_camera.get_image()
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        image_drawn = img.copy()
        kernel = np.ones((3,3))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (9, 9), cv.BORDER_DEFAULT)
        lower, upper = auto_limit(gblur, 0.33)
        thresh = cv.threshold(gblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        erode = cv.erode(adaptive, np.ones((5,5)))
        opening = cv.dilate(erode, np.ones((5,5)))
        dilated = cv.dilate(opening, np.ones((3,3)))
        closing = cv.erode(dilated, np.ones((3,3)))
        
        canny = cv.Canny(gray, lower, upper)
        
        results = pytesseract.image_to_data(adaptive, config = custom_config, output_type=pytesseract.Output.DICT, lang = 'eng')
        n_boxes = len(results['level'])

        for i in range(n_boxes):
            (x, y, w, h) = (results['left'][i], results['top'][i], results['width'][i], results['height'][i])
            confidence = int(results['conf'][i])
            if confidence > 10:
                cv.rectangle(image_drawn, (x, y), (x + w, y + h), (255, 0, 0), 1)
                cv.putText(image_drawn, results['text'][i], (x, y+20), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # h_img, w_img, c = img.shape
        # boxes = pytesseract.image_to_boxes(canny)
        # #print('norm:',boxes)
        # for boxes in boxes.splitlines():
        #     #print('splitline:',boxes)
        #     boxes = boxes.split(' ')
        #     #print('split:',boxes)
        #     x,y,w,h = int(boxes[1]), int(boxes[2]), int(boxes[3]), int(boxes[4])
        #     #print(x,y,w,h)
        #     #cv.rectangle(image_drawn, (x, y), (x + w, y + h), (0, 0, 255), 1)
        #     cv.rectangle(image_drawn, (x, h_img-y), (w, h_img - h), (0, 255, 0), 1)
        #     cv.putText(image_drawn, d['text'][i], (x, h_img-y+20), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv.imshow('img', img)
        cv.imshow('result', image_drawn)
        cv.imshow('canny', canny)
        cv.imshow('opening', opening)
        cv.imshow('closing', closing)
        cv.imshow('adaptive', adaptive)
        #cv.imshow('thresh',thresh)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def matplot():
    # DO not use: crashing 
    import keras_ocr
    from matplotlib import pyplot as plt
    pipeline = keras_ocr.pipeline.Pipeline()

    while True:
        img = my_camera.get_image()
        images = [keras_ocr.tools.read(url) for url in [
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/1.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/2.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/3.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/4.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/5.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/6.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/7.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/8.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/9.jpg',
            'https://raw.githubusercontent.com/Thangasami/OCR-/main/number/10.jpg'
            ]
        ]
        print('start')
        prediction_groups = pipeline.recognize(images)
        print('hi')
        fig, axs = plt.subplot(nrows=len(images),figsize=(10,10))
        print('dfs')
        for ax, image, prediction in zip(axs,images,prediction_groups):
            keras_ocr.tools.drawAnnotations(image=image, prediction=prediction, ax=ax)
        print('end')
        cv.imshow('result', img)
        cv.waitKey(0)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

def pics():
    while True:
        img = my_camera.get_image()
        image_drawn = img.copy()
        kernel = np.ones((3,3))
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gblur = cv.GaussianBlur(gray, (7, 7), cv.BORDER_DEFAULT)
        lower, upper = auto_limit(gblur, 0.33)

        thresh = cv.threshold(gblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
        adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
        dilated = cv.dilate(adaptive, kernel)
        opening = cv.erode(dilated, kernel)
        erode = cv.erode(opening, kernel)
        closing = cv.dilate(erode, kernel)

        canny = cv.Canny(closing, lower, upper)
        
        cv.imshow('result', image_drawn)
        cv.imshow('canny', canny)
        cv.imshow('adaptive', adaptive)
        cv.imshow('opening', opening)
        cv.imshow('closing', closing)
        cv.imshow('thresh',thresh)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    #my_camera = camera.Camera2()
    #img = cv.imread('/home/mike/cobot/scripts/classes/MTP/abc.jpg')
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #
    #tesseract()
    #cv.destroyAllWindows()
    a = -4
    print(a)
    a = int(a)
    print(a)