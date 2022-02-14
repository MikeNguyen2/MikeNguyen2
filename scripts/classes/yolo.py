import requests
import base64
from PIL import Image
import cv2 as cv
import numpy as np
import camera
# Load Input Stream
cv.namedWindow("preview")
#video = cv.VideoCapture(0)#, cv.CAP_V4L2)

my_camera = camera.Camera2()
# Construct the URL
upload_url = "".join([
    "http://127.0.0.1:9001/labor-mtp-xmy9z/1",
    "?api_key=dF1uuuGTVmDrc9Vfsm7c",
    "&format=image",
    "&stroke=5"
])


def infer():
    # Get the current image from the webcam
    #ret, img = video.read()
    img = my_camera.get_image()
    print(img.shape)

    # Resize (while maintaining the aspect ratio) to improve speed and save bandwidth
    height, width, channels = img.shape
    scale = 416.0 / max(height, width)
    img = cv.resize(img, (int(np.round(scale * width)), int(np.round(scale * height))))

    # Encode image to base64 string
    retval, buffer = cv.imencode('.jpg', img)
    img_str = base64.b64encode(buffer)

    #print(img_str)
    # Get prediction from Roboflow Infer API
    resp = requests.post(upload_url, data=img_str, headers={
        "Content-Type": "application/x-www-form-urlencoded"
    }, stream=True).raw
    print(bytearray(resp.read()))
    # Parse result image
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    print(image)
    image = cv.imdecode(image, cv.IMREAD_COLOR)
    return image
while 1:
    # Synchronously get a prediction from the Roboflow Infer API
    image = infer()
    # And display the inference results
    #print(image)
    cv.imshow('image', image)
    # On "q" keypress, exit
    if(cv.waitKey(1) == ord('q')):
        break
video.release()
cv.destroyAllWindows()