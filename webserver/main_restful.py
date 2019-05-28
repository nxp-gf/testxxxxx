#encoding=utf-8
#!/usr/bin/env python
#curl -X GET  "http://0.0.0.0:8200/inference?solution=face-recognition&action=inference" -F "image=@Pekinese_347.jpg"
from flask import Flask
from flask_restful import reqparse, Api, Resource, request
import werkzeug
from io import BytesIO
import numpy as np
from PIL import Image
from inference import FaceRecognition,ImageClassification,ObjectDetection
#import cv2

FaceRecognition.init("models/face_recognition")
ImageClassification.init("models/image_classification")
ObjectDetection.init("models/object_detection")

parser = reqparse.RequestParser()

app = Flask(__name__)
api = Api(app)
parser.add_argument('solution', type=str, location='args')
parser.add_argument('action', type=str, location='args')
parser.add_argument('name', type=str, location='args')
parser.add_argument('image', type=werkzeug.datastructures.FileStorage, location='files')

class Inference(Resource):
    def get(self):
        args = parser.parse_args()
        print(args)

        pif=BytesIO()
        args["image"].save(pif)
        pif.seek(0)
        pilimg = Image.open(pif).convert('RGB')
        img = np.array(pilimg)
        img = img[:, :, ::-1].copy() 
        #npimg = np.fromstring(args['image'].read(), np.uint8)
        #img = cv2.imdecode(npimg, cv2.CV_LOAD_IMAGE_COLOR)
        print(img.shape)
        image_char = img.astype(np.uint8).tostring()

        ret = ""
        if args['solution'] == 'image-classification':
            ret = ImageClassification.recognize(img.shape[0], img.shape[1], image_char)
        elif args['solution'] == 'object-detection':
            ret = ObjectDetection.recognize(img.shape[0], img.shape[1], image_char)
        elif args['solution'] == 'face-recognition':
            if args['action'] == 'add':
                ret = FaceRecognition.recognize(img.shape[0], img.shape[1], image_char, args['name'])
            elif args['action'] == 'delete':
                ret = FaceRecognition.delete(args['name'])
                pass
            elif args['action'] == 'recognize':
                ret = FaceRecognition.recognize(img.shape[0], img.shape[1], image_char, "")

        return ret

##
## Actually setup the Api resource routing here
##
api.add_resource(Inference, '/inference')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8200, threaded=False)
