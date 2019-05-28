/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#include "bitmap_helpers.h"
#include "get_top_n.h"

#include "MTCNN-light/mtcnn.h"
#include "MTCNN-light/mropencv.h"
#include "face-db/featuredb.h"

#define LOG(x) std::cerr

#include "boost/python.hpp"

namespace bp = boost::python;

namespace tflite {
namespace tflite_inference {

FeatureDB *featuredb;
mtcnn *detector;


double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

int Delete(bp::str str){
  const char *name = ((const char *) bp::extract<const char *>(str));
  return featuredb->del_feature(name);
}

static std::unique_ptr<tflite::FlatBufferModel> model;
static std::unique_ptr<tflite::Interpreter> interpreter;

void InitInference(bp::str inpath) {
  std::string model_path = bp::extract<std::string>(inpath);

  //const char *model_name = ((const char *) bp::extract<const char *>(inpath));
  featuredb = new FeatureDB(model_path, 0.70);
  detector = new mtcnn(model_path + "/mtcnn");
  //detector = new mtcnn(std::string(model_path));

  // 1. create model
  model = tflite::FlatBufferModel::BuildFromFile((model_path + "/model.tflite").c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << model_path << "\n";
    exit(-1);
  }
  model->error_reporter();

  // 2. create OpResolver
  tflite::ops::builtin::BuiltinOpResolver resolver;

  // 3. create Interpreter
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  // 4. setting parameter for Interpreter
  interpreter->UseNNAPI(false); // using NNAPI for accel?
  interpreter->SetAllowFp16PrecisionForFp32(false); // set date format

  interpreter->SetNumThreads(4); // set running threads number

  // 8. alloc inputs/outputs tensor memory
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

}

float *RunInference(std::vector<uint8_t> &in, int image_width, int image_height, int image_channels) {

  // 6. get input tensor index
  int input = interpreter->inputs()[0];

  // 7. get inputs/outputs tensor index array
  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();

  // 9. get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];

  // 10. resize input
  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, true);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, false);
      break;
    default:
      LOG(FATAL) << "cannot handle input type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }

  // 11. profiling??
  profiling::Profiler* profiler = new profiling::Profiler();
  interpreter->SetProfiler(profiler);

  // 12. start Invoke
  if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
  }

  float *features = interpreter->typed_output_tensor<float>(0);
  return features;
}

bp::dict Recognize(int rows,int cols,bp::str img_data, bp::str name_data)
{
    unsigned char *data = (unsigned char *) ((const char *) bp::extract<const char *>(img_data));
    cv::Mat image= cv::Mat(rows, cols, CV_8UC3,data);
    const char *namein = ((const char *) bp::extract<const char *>(name_data));
    bp::dict res;

    int height,width,channels;
    cv::Mat imageOrig = image.clone();

    std::vector<FaceInfo>fds;
    detector->Detect(image,fds);

    if(fds.size()) {
        FaceInfo &info = fds.at(0);

        cv::Rect select;
        cv::Mat headImage;
        select.x      = info.bbox.y;
        select.y      = info.bbox.x;
        select.width  = info.bbox.width;
        select.height = info.bbox.height;

        headImage = imageOrig(select);

        height = headImage.size().width;
        width = headImage.size().height;
        channels = headImage.channels();

        std::vector<uint8_t> output(height * width * channels);
        for(int i = 0;i < height ;i++){
            Vec3b *data = headImage.ptr<Vec3b>(i);
            for(int j = 0;j < width ;j++){
                int dst_pos = (i * width + j) * channels;
                output[dst_pos + 2] = data[j][0] ;
                output[dst_pos + 1] = data[j][1];
                output[dst_pos + 0] = data[j][2];
            }
        }
        float *tmp = RunInference(output, height, width, channels);
        std::vector<float> feature(tmp, tmp + 512);
        if (strlen(namein) == 0) {
            std::string name = featuredb->find_name(feature);
            res["name"] = name;
            res["x"] = select.x;
            res["y"] = select.y;
            res["w"] = select.width;
            res["h"] = select.height;
        } else {
            featuredb->add_feature(namein, feature);
        }
    }
    return res;
}

}  // namespace tflite_inference
}  // namespace tflite


BOOST_PYTHON_MODULE(FaceRecognition)
{
    def("init", tflite::tflite_inference::InitInference);
    def("delete", tflite::tflite_inference::Delete);
    def("recognize", tflite::tflite_inference::Recognize);
}
