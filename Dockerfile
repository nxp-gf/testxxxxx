FROM eiqcloud/tflite-inference
COPY qemu-aarch64-static /usr/bin/
COPY webserver /
EXPOSE 8000
EXPOSE 8200
