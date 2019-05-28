#ifndef MTCNN_H
#define MTCNN_H
#include "network.h"
#include "FaceInfo.h"

//using namespace facedetecion;
static std::string modeldir;
class Pnet
{
public:
    Pnet();
    ~Pnet();
    void run(Mat &image, float scale);

    float nms_threshold;
    mydataFmt Pthreshold;
    bool firstFlag;
    vector<struct Bbox> boundingBox_;
    vector<orderScore> bboxScore_;
private:
    //the image for mxnet conv
    pBox *rgb;
    pBox *conv1_matrix;
    //the 1th layer's out conv
    pBox *conv1;
    pBox *maxPooling1;
    pBox *maxPooling_matrix;
    //the 3th layer's out
    pBox *conv2;
    pBox *conv3_matrix;
    //the 4th layer's out   out
    pBox *conv3;
    pBox *score_matrix;
    //the 4th layer's out   out
    pBox *score_;
    //the 4th layer's out   out
    pBox *location_matrix;
    pBox *location_;

    //Weight
    Weight *conv1_wb;
    pRelu *prelu_gmma1;
    Weight *conv2_wb;
    pRelu *prelu_gmma2;
    Weight *conv3_wb;
    pRelu *prelu_gmma3;
    Weight *conv4c1_wb;
    Weight *conv4c2_wb;

    void generateBbox(const pBox *score, const pBox *location, mydataFmt scale);
};

class Rnet
{
public:
    Rnet(std::string modeldir);
    ~Rnet();
    float Rthreshold;
    void run(Mat &image);
    pBox *score_;
    pBox *location_;
private:
    pBox *rgb;

    pBox *conv1_matrix;
    pBox *conv1_out;
    pBox *pooling1_out;

    pBox *conv2_matrix;
    pBox *conv2_out;
    pBox *pooling2_out;

    pBox *conv3_matrix;
    pBox *conv3_out;

    pBox *fc4_out;
    
    //Weight
    Weight *conv1_wb;
    pRelu *prelu_gmma1;
    Weight *conv2_wb;
    pRelu *prelu_gmma2;
    Weight *conv3_wb;
    pRelu *prelu_gmma3;
    Weight *fc4_wb;
    pRelu *prelu_gmma4;
    Weight *score_wb;
    Weight *location_wb;

    void RnetImage2MatrixInit(pBox *pbox);
};

class Onet
{
public:
    Onet(std::string modeldir);
    ~Onet();
    void run(Mat &image);
    float Othreshold;
    pBox *score_;
    pBox *location_;
    pBox *keyPoint_;
private:
    pBox *rgb;
    pBox *conv1_matrix;
    pBox *conv1_out;
    pBox *pooling1_out;

    pBox *conv2_matrix;
    pBox *conv2_out;
    pBox *pooling2_out;

    pBox *conv3_matrix;
    pBox *conv3_out;
    pBox *pooling3_out;

    pBox *conv4_matrix;
    pBox *conv4_out;

    pBox *fc5_out;

    //Weight
    Weight *conv1_wb;
    pRelu *prelu_gmma1;
    Weight *conv2_wb;
    pRelu *prelu_gmma2;
    Weight *conv3_wb;
    pRelu *prelu_gmma3;
    Weight *conv4_wb;
    pRelu *prelu_gmma4;
    Weight *fc5_wb;
    pRelu *prelu_gmma5;
    Weight *score_wb;
    Weight *location_wb;
    Weight *keyPoint_wb;
    void OnetImage2MatrixInit(pBox *pbox);
};

class mtcnn
{
public:
    mtcnn(std::string dir="model");
    ~mtcnn();
    int Init();
    int SetMinFaceSize(int size = 40);
    FaceDetectionResult Detect(Mat &image,std::vector<FaceInfo>&faces);
    int Release();
private:
    int minsize = 100;
    int row=480;
    int col=640;
    Mat reImage;
    float nms_threshold[3];
    vector<float> scales_;
    Pnet *simpleFace_;
    vector<struct Bbox> firstBbox_;
    vector<struct orderScore> firstOrderScore_;
    Rnet refineNet;
    vector<struct Bbox> secondBbox_;
    vector<struct orderScore> secondBboxScore_;
    Onet outNet;
    vector<struct Bbox> thirdBbox_;
    vector<struct orderScore> thirdBboxScore_;
};

#endif
