#!/bin/bash
echo "Compiling"

cd face_recognition/
make
cd image_classification/
make
cd object_detection/
make

cp face_recognition/FaceRecognition.so webserver/inference/
cp image_classification/ImageClassification.so webserver/inference/
cp object_detection/ObjectDetection.so webserver/inference/

cp -a webserver target
