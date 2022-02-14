import cv2 as cv
import numpy as np
import pytesseract
from pytesseract import Output
import camera
#import pyautogui

my_camera = camera.Camera2()

def auto_limit(image, sigma):
    median = np.median(image)
    lower_limit = int(max(0, (1 - sigma) * median))
    upper_limit = int(min(255, (1 + sigma) * median))
    return lower_limit, upper_limit

#Load input image, convert from BGR to RGB ch ordering, and
# use Tesseract to localize each area of text in the input image
custom_config = r'--oem 3 --psm 11'
image = cv.imread('/home/mike/cobot/scripts/classes/MTP/lab.png')

while True:
    kernel = np.ones((3,3))
    
    #image = my_camera.get_image()
    rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
    gblur = cv.GaussianBlur(gray, (9, 9), cv.BORDER_DEFAULT)
    lower, upper = auto_limit(gblur, 0.33)
    thresh = cv.threshold(gblur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)[1]
    adaptive = cv.adaptiveThreshold(gblur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,11,2)
    erode = cv.erode(adaptive, np.ones((5,5)))
    opening = cv.dilate(erode, np.ones((5,5)))
    dilated = cv.dilate(opening, np.ones((3,3)))
    closing = cv.erode(dilated, np.ones((3,3)))
    canny = cv.Canny(gray, lower, upper)
    
    counter = 0
    for img in [canny]:#[rgb, gray, thresh, adaptive, opening, canny]:
        image_drawn = image.copy()
        results = pytesseract.image_to_data(img, output_type=Output.DICT, config=custom_config)
            # 'image_to_data' detects and localizes text 

        #Loop over each indiv text localizations
        for i in range(0,  len(results["text"] )  ):
            #extract bounding box coordinates of the text region from the current result
            x = results["left"][i]
            y = results["top"][i]
            w = results["width"][i]
            h = results["height"][i]

            #extract OCR itself along with conf of text localztn
            text = results["text"][i]
            #print(results["conf"][i])
            conf = int( results["conf"][i] )


        #Filter out weak conf text localztns
            if conf > 10:

                #display conf and text to terminal
                # print("Confidence: {}".format(conf) )
                # print("Text: {}".format(text) )
                # print("")

                #remove non-ASCII text so we can draw text on image using OpenCV, then draw bounding box around text with text itself
                text = "".join( [c if ord(c) < 128 else "" for c in text] ).strip()
                cv.rectangle(image_drawn,  (x,y),  (x+w, y+h),  (0, 255, 0), 2 )
                cv.putText(image_drawn, text, (x, y+20), cv.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        #Show output image
        #image_drawn = cv.resize(image_drawn,(300,175))
        cv.imshow("Image"+str(counter), image_drawn)
        counter+=1
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    #  