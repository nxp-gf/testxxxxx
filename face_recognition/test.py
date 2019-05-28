import FaceRecognition
import numpy as np

import cv2

img = cv2.imread("../test_images/Ai_Sugiyama_0001.bmp")
image_char = img.astype(np.uint8).tostring()

FaceRecognition.init("./model")
#facerecognition.recognition(1)
#facerecognition.recognition(0)
ret = FaceRecognition.recognize(img.shape[0], img.shape[1], image_char, "")
print(ret)
