//c++ network  author : liqi
//Nangjing University of Posts and Telecommunications
//date 2017.5.21,20:27
#ifndef NETWORK_H
#define NETWORK_H
#include "opencv2/opencv.hpp"
#include <algorithm>
#include <stdlib.h>
#include <memory.h>
#include <fstream>
#include <cstring>
//#include <openBLAS/cblas.h>
#include <cblas.h>
#include <string>
#include <math.h>
#include "pBox.h"
using namespace cv;

void addbias(pBox *pbox, mydataFmt *pbias);
void image2Matrix(const Mat &image, const pBox *pbox);
void featurePad(const pBox *pbox, const pBox *outpBox, const int pad);
void feature2Matrix(const pBox *pbox, pBox *Matrix, const Weight *weight);
void maxPooling(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);
void relu(pBox *pbox, mydataFmt *pbias);
void prelu(pBox *pbox, mydataFmt *pbias, mydataFmt *prelu_gmma);
void convolution(const Weight *weight, const pBox *pbox, pBox *outpBox, const pBox *matrix);
void fullconnect(const Weight *weight, const pBox *pbox, pBox *outpBox);
void readData(string filename, long dataNumber[], mydataFmt *pTeam[]);
long initConvAndFc(Weight *weight, int schannel, int lchannel, int kersize, int stride, int pad);
void initpRelu(pRelu *prelu, int width);
void softmax(const pBox *pbox);

void image2MatrixInit(Mat &image, pBox *pbox);
void featurePadInit(const pBox *pbox, pBox *outpBox, const int pad);
void maxPoolingInit(const pBox *pbox, pBox *Matrix, int kernelSize, int stride);
void feature2MatrixInit(const pBox *pbox, pBox *Matrix, const Weight *weight);
void convolutionInit(const Weight *weight, const pBox *pbox, pBox *outpBox, const pBox *matrix);
void fullconnectInit(const Weight *weight, pBox *outpBox);


bool cmpScore(struct orderScore lsh, struct orderScore rsh);
void nms(vector<struct Bbox> &boundingBox_, vector<struct orderScore> &bboxScore_, const float overlap_threshold, string modelname = "Union");
void refineAndSquareBbox(vector<struct Bbox> &vecBbox, const int &height, const int &width);

#endif
