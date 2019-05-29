import cv2
from base_camera import BaseCamera
import Queue
import numpy as np
import time,os,threading
import json
import facerecognition

facerecognition.init("model.tflite")

def rotate(image, angle, center=None, scale=1.0):
    (h, w) = image.shape[:2]

    if center is None:
        center = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    return rotated

class Camera(BaseCamera):
    video_source = []
    reg_ret = []
    solution = ""
    new_name = None

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def set_solution(solution):
        Camera.solution = solution

    @staticmethod
    def del_person(name):
        facerecognition.delete(name)

    @staticmethod
    def add_person(name):
        Camera.new_name = name

    @staticmethod
    def object_detection():
        f = os.popen("object-detection/object-detection -t 4 -l labels.txt -m model.tflite -i tmp.bmp")
        output = f.readlines()
        rets = []
        for line in output:
            rets.append(json.loads(line))
        return rets

    @staticmethod
    def image_classification():
        f = os.popen("image-classification/image-classification -t 4 -l labels.txt -m model.tflite -i tmp.bmp")
        output = f.readlines()
        return output

    @staticmethod
    def frames():
        framequeue = []
        cap = cv2.VideoCapture(Camera.video_source)
        if not cap.isOpened():
            raise RuntimeError('Could not start camera. Index:' , i)

        cap_lock = threading.Lock()

        def drop_img_thread():
            while True:
                cap_lock.acquire()
                ret, img = cap.read()
                cap_lock.release()
                time.sleep(0.05)

        threading.Thread(target = drop_img_thread).start()

        while True:
            cap_lock.acquire()
            ret, img = cap.read()
            cap_lock.release() 
 
            if img is None:
                continue

            if Camera.solution == "object_detection":
                cv2.imwrite("./tmp.bmp", img)
                rets = Camera.object_detection()
                for ret in rets:
                    if ret['scores'] > 0.5:
                        x1, y1, x2, y2 = ret['rect1'] * 6.4, ret['rect0'] * 4.8, ret['rect3'] * 6.4, ret['rect2'] * 4.8
                        cv2.rectangle(img,(int(x1), int(y1)),(int(x2), int(y2)),(127,255,0),1)
                        cv2.putText(img, ret['classes'],(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX, 0.6,(242,243,231),2)
            elif Camera.solution == "image_classification":
                cv2.imwrite("./tmp.bmp", img)
                ret = Camera.image_classification()
                if int(ret[1].split(":")[0]) < 65:
                    ret[1] = "NULL"
                cv2.putText(img, ret[1],(10,30),cv2.FONT_HERSHEY_COMPLEX, 0.6,(242,243,231),2)
            else:
                image_char = img.astype(np.uint8).tostring()
                if Camera.new_name != None:
                    ret = facerecognition.recognize(img.shape[0], img.shape[1], image_char, Camera.new_name)
                    Camera.new_name = None
                else:
                    ret = facerecognition.recognize(img.shape[0], img.shape[1], image_char, "")
                if 'name' in ret.keys():
                    x1, y1, x2, y2 = ret['x'], ret['y'], ret['x'] + ret['w'], ret['y'] + ret['h']
                    cv2.rectangle(img,(int(x1), int(y1)),(int(x2), int(y2)),(127,255,0),1)
                    cv2.putText(img, ret['name'],(int(x1),int(y1)),cv2.FONT_HERSHEY_COMPLEX, 0.6,(242,243,231),2)
                print(ret)
                pass

            yield cv2.imencode('.png', img)[1].tobytes()
