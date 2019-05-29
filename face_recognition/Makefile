
INC = -I./ -I../common -I/usr/include/python2.7/
CFLAGS = -std=c++11
LIBS = -L./ -ltensorflow-lite -lpthread -ldl -lblas -lboost_python\
       -lopencv_core -lopencv_highgui -lopencv_video -lopencv_imgproc
CPP = g++
RM  = rm -f
SRC = face_recognition.cc ../common/bitmap_helpers.cc \
      MTCNN-light/mtcnn.cpp  \
      MTCNN-light/network.cpp \
      MTCNN-light/pBox.cpp \
      face-db/featuredb.cpp
OUTPUT = FaceRecognition

all:
	$(CPP) $(CFLAGS) -fPIC $(INC) -o $(OUTPUT).so -shared $(SRC)  $(LIBS) 
clean:
	$(RM) $(OUTPUT).so
