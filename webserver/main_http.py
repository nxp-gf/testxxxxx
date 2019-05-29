#!/usr/bin/env python
from importlib import import_module
import os,time
from flask import Flask, render_template, Response
import argparse
from flask_socketio import SocketIO,emit
from camera_opencv import *

os.environ["CAP_PROP_FRAME_WIDTH"] = "640"
os.environ["CAP_PROP_FRAME_HEIGHT"] = "480"


parser = argparse.ArgumentParser()
parser.add_argument('--dev', type=str, default="defalut",
                    help='[usb|"url" of IP camera]input video device')
parser.add_argument('--httpport', type=int,
                    help='The port for http server')
parser.add_argument('--svr', type=str,
                    help='The ip for training server')
args = parser.parse_args()

print("Initialzing face recognition engine.")
if 'usb' in args.dev:
    dev = int(args.dev[3:])
    print("Using onboard usb camera, ", dev)
else:
    #dev = args.dev
    dev = "rtsp://10.193.20.163:554/user=admin_password=6QNMIQGe_channel=1_stream=1.sdp?real_stream"
    print("Using ip camera with url(s)", dev)
Camera.set_video_source(dev)


if args.httpport != None:
    HTTP_PORT = args.httpport
else:
    HTTP_PORT = 5000

fdir = os.path.dirname(os.path.realpath(__file__))

app = Flask(__name__)
socketio = SocketIO(app)
#socketio.init_app(app)

@app.route('/face-recognition')
def index_face():
    """Video streaming home page."""
    
    Camera.set_solution("face_recognition")
    return render_template('index_face.html')

@app.route('/object-detection')
def index_object():
    """Video streaming home page."""
    Camera.set_solution("object_detection")
    return render_template('index_web.html')

@app.route('/image-classification')
def index_image():
    """Video streaming home page."""
    Camera.set_solution("image_classification")
    return render_template('index_web.html')

def gen(camera):
    """Video streaming generator function."""
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/png\r\n\r\n' + frame + b'\r\n')

@app.route('/videoel')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if args.dev == 'laptop':
        return Response()
    else:
        print("video_feed")
        return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('request',namespace='/testnamespace')
def give_response(data):
    msg_type = data.get('type')
    msg_data = data.get('data')

    print "in give_response"
    if (msg_type == "ADDPERSON_REQ"):
        print "TRAINSTART_REQ"
        Camera.add_person(str(msg_data))
    elif (msg_type == "DELPERSON_REQ"):
        print "DELPERSON_REQ"
        Camera.del_person(str(msg_data))


@socketio.on('connect', namespace='/testnamespace')
def test_connect():
    print("in test_connect")

if __name__ == '__main__':
    socketio.run(app,debug=True,host='0.0.0.0',use_reloader=False,port=8000)
