import ObjectDetection
import numpy as np

import cv2

img = cv2.imread("../test_images/Pekinese_347.jpg")
image_char = img.astype(np.uint8).tostring()

ObjectDetection.init("./model")
ret = ObjectDetection.recognize(img.shape[0], img.shape[1], image_char)
print(ret)
