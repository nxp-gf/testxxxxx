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

#ifndef TENSORFLOW_LITE_INFERENCE_BITMAP_HELPERS_H_
#define TENSORFLOW_LITE_INFERENCE_BITMAP_HELPERS_H_

#include "bitmap_helpers_impl.h"

namespace tflite {
namespace tflite_inference {

std::vector<uint8_t> read_bmp(const std::string& input_bmp_name, int* width,
                              int* height, int* channels);

template <class T>
void resize(T* out, uint8_t* in, int image_height, int image_width,
            int image_channels, int wanted_height, int wanted_width,
            int wanted_channels, bool input_floating);

// explicit instantiation
template void resize<uint8_t>(uint8_t*, unsigned char*, int, int, int, int, int,
                              int, bool);
template void resize<float>(float*, unsigned char*, int, int, int, int, int,
                            int, bool);

}  // namespace tflite_inference
}  // namespace tflite

#endif  // TENSORFLOW_LITE_INFERENCE_BITMAP_HELPERS_H_
