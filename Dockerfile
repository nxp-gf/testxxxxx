FROM eiqcloud/tflite-inference
COPY target /
EXPOSE 8000
EXPOSE 8200
