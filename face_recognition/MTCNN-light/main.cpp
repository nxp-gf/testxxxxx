#include "mtcnn.h"
#include <time.h>
#include "mropencv.h"
#include <sys/stat.h>

int mtcnnCut(cv::Mat &image, std::vector<FaceInfo> &fds)
{
	cv::Rect select;
	cv::Mat headImage;
	
	for(int i=0; i<fds.size(); i++) {
		FaceInfo &info = fds.at(i);
		select.x      = info.bbox.y;
		select.y      = info.bbox.x;
		select.width  = info.bbox.width;
		select.height = info.bbox.height;
		
		headImage = image(select);

		std::vector<unsigned char> buff;
		cv::imencode(".bmp", headImage, buff);

		cout << buff.size() << endl;
		cout << headImage.size().width << endl;
		cout << headImage.size().height << endl;
		cout << headImage.channels() << endl;
		
		std::ostringstream ostr;
		ostr << "results/head_" << i << ".bmp";

		if(!imwrite(ostr.str().c_str(), headImage))
			cout << "failed!\n" << endl;
	}
	
	return 0;
}

int mtcnnDetect(cv::Mat &image)
{
	if (!image.data)
		return -1;

	cv::Mat imageOrig = image.clone();

	static mtcnn find;
//	find.SetMinFaceSize(60);
	TickMeter tm;
	tm.start();
    std::vector<FaceInfo>fds;
	find.Detect(image,fds);
	tm.stop();
	cout << tm.getTimeMilli() << "ms" << endl;

	mtcnnCut(imageOrig, fds);
	
	return 0;
}

int testimage(const string imgpath)
{
	Mat image = cv::imread(imgpath);
	mtcnnDetect(image);
    imwrite("results/result.jpg",image);
	return 0;
}

int testcamera(int device=0)
{
	Mat image;
	VideoCapture cap(device);
	if (!cap.isOpened())
		cout << "fail to open!" << endl;
	while (1){
		cap >> image;
		if (!image.data)
			break;
		mtcnnDetect(image);
		imshow("mtcnn", image);
		if (waitKey(1) >= 0) break;
	}
    waitKey(0);
    image.release();
    return 0;
}

int deleteResults(char *strDir)
{
	char cmd[128] = {0};
	sprintf(cmd, "rm %s/*", strDir); 
	system(cmd);
	return 0;
}

int main(int argc, char* argv[])
{
	if(argc < 2) {
		cout << "Please input one test image!" << endl;
		return -1;
	}

	deleteResults("./results");

	testimage(argv[1]);
	
//	testcamera();

	return 0;
}
