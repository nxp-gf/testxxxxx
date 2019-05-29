import ImageClassification 
import numpy as np

import cv2

img = cv2.imread("./tmp.jpg")
image_char = img.astype(np.uint8).tostring()

ImageClassification.init("./model")
ret = ImageClassification.recognize(img.shape[0], img.shape[1], image_char)
print(ret)
