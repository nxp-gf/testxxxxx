#!/bin/bash
echo "Compiling"
ROOT=`pwd`
TARGET=$1

cd $ROOT/face_recognition/
make
cd $ROOT/image_classification/
make
cd $ROOT/object_detection/
make

cd $ROOT
cp face_recognition/FaceRecognition.so webserver/inference/
cp image_classification/ImageClassification.so webserver/inference/
cp object_detection/ObjectDetection.so webserver/inference/

cp -a webserver $TARGET
