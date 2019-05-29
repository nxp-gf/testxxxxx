
INC = -I./ -I../common -I/usr/include/python2.7/
CFLAGS = -std=c++11
LIBS = -L./ -ltensorflow-lite -lpthread -ldl -lblas -lboost_python\
       -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc
CPP = g++
RM  = rm -f
SRC = object_detection.cc ../common/bitmap_helpers.cc
OUTPUT = ObjectDetection

all:
	$(CPP) $(CFLAGS) -fPIC $(INC) -o $(OUTPUT).so -shared $(SRC)  $(LIBS) 
clean:
	$(RM) $(OUTPUT).so
