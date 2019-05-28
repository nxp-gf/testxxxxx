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
#include "opencv2/opencv.hpp"

#include "bitmap_helpers.h"
#include "get_top_n.h"

#define LOG(x) std::cerr

#include "boost/python.hpp"

namespace bp = boost::python;
using namespace cv;

namespace tflite {
namespace tflite_inference {

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

static std::unique_ptr<tflite::FlatBufferModel> model;
static std::unique_ptr<tflite::Interpreter> interpreter;

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}


std::vector<string> labels;
size_t label_count;

void InitInference(bp::str inpath) {
  std::string model_path = bp::extract<std::string>(inpath);

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

  if (ReadLabelsFile(model_path + "/labels.txt", &labels, &label_count) != kTfLiteOk)
    exit(-1);
}

std::vector<std::pair<float, int>> RunInference(std::vector<uint8_t> &in, int image_width, int image_height, int image_channels) {
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

  // 11. profiling
  profiling::Profiler* profiler = new profiling::Profiler();
  interpreter->SetProfiler(profiler);

  // 12. start Invoke
  if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
  }

  const float threshold = 0.001f;

  std::vector<std::pair<float, int>> top_results;

  // 13. get output top n
  int output = interpreter->outputs()[0];
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto output_size = output_dims->data[output_dims->size - 1];
  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(interpreter->typed_output_tensor<float>(0), output_size,
                       5, threshold, &top_results, true);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(interpreter->typed_output_tensor<uint8_t>(0),
                         output_size, 5, threshold,
                         &top_results, false);
      break;
    default:
      LOG(FATAL) << "cannot handle output type "
                 << interpreter->tensor(input)->type << " yet";
      exit(-1);
  }


  return top_results;
}

bp::list Recognize(int rows,int cols,bp::str img_data)
{
    unsigned char *data = (unsigned char *) ((const char *) bp::extract<const char *>(img_data));
    cv::Mat image= cv::Mat(rows, cols, CV_8UC3,data);
    bp::list recog_result;

    int height,width,channels;
    height = image.size().width;
    width = image.size().height;
    channels = image.channels();

    std::vector<uint8_t> output(height * width * channels);
    for(int i = 0;i < height ;i++){
        Vec3b *data = image.ptr<Vec3b>(i);
        for(int j = 0;j < width ;j++){
            int dst_pos = (i * width + j) * channels;
            output[dst_pos + 2] = data[j][0] ;
            output[dst_pos + 1] = data[j][1];
            output[dst_pos + 0] = data[j][2];
        }
    }
    std::vector<std::pair<float, int>> top_results = RunInference(output, width, height, channels);
    for (const auto& result : top_results) {
        bp::dict res;
        const int confidence = (int) (result.first * 100);
        const int index = result.second;
        res["confidence"] = confidence;
        res["label"] = labels[index];
        recog_result.append(res);
    }

    return recog_result;
}

}  // namespace tflite_inference
}  // namespace tflite


BOOST_PYTHON_MODULE(ImageClassification)
{
    def("init", tflite::tflite_inference::InitInference);
    def("recognize", tflite::tflite_inference::Recognize);
}

