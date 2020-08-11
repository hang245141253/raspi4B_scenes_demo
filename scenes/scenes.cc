// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.



#include "paddle_api.h"
#include <arm_neon.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <vector>
#include <limits>

using namespace paddle::lite_api;  // NOLINT

const std::vector<float> INPUT_MEAN = {0.f,0.f,0.f};
const std::vector<float> INPUT_STD = {1.f,1.f,1.f};

int64_t ShapeProduction(const shape_t& shape) {
  int64_t res = 1;
  for (auto i : shape) res *= i;
  return res;
}

std::vector<std::string> load_labels(const std::string &path) {
  std::ifstream file;
  std::vector<std::string> labels;
  file.open(path);
  while (file) {
    std::string line;
    std::getline(file, line);
    labels.push_back(line);
  }
  file.clear();
  file.close();
  return labels;
}

void RunModel(cv::Mat &photo,
              std::vector<std::string> &word_labels,
              std::shared_ptr<paddle::lite_api::PaddlePredictor> &predictor) {

    //Pre_process
    cv::cvtColor(photo, photo, CV_BGR2RGB);//BGR->RGB 与训练时输入一致


//    std::cout << photo << std::endl;//查看形状
//    cv::imshow("num", photo);
//    cv::waitKey(0);

      std::cout << "当前图像尺寸为:"
                << photo.cols
                << "x"
                << photo.rows
                << std::endl;
    if(photo.cols != 224 || photo.rows != 224) {
        cv::resize(photo, photo, cv::Size(224, 224), 0.f, 0.f);
        std::cout << "img has been resize."
                  << std::endl;
    }

    photo.convertTo(photo, CV_32FC1, 1.f / 255.f , 0.f);//归一化


    // Get Input Tensor
    std::unique_ptr<Tensor> input_tensor(std::move(predictor->GetInput(0)));
    input_tensor->Resize({1, 3, 224, 224});
    auto* input_data = input_tensor->mutable_data<float>();

    // NHWC->NCHW
    int image_size = photo.cols*photo.rows;
    const float *image_data = reinterpret_cast<const float *>(photo.data);
    float32x4_t vmean0 = vdupq_n_f32(INPUT_MEAN[0]);
    float32x4_t vmean1 = vdupq_n_f32(INPUT_MEAN[1]);
    float32x4_t vmean2 = vdupq_n_f32(INPUT_MEAN[2]);
    float32x4_t vscale0 = vdupq_n_f32(1.0f / INPUT_STD[0]);
    float32x4_t vscale1 = vdupq_n_f32(1.0f / INPUT_STD[1]);
    float32x4_t vscale2 = vdupq_n_f32(1.0f / INPUT_STD[2]);
    float *input_data_c0 = input_data;
    float *input_data_c1 = input_data + image_size;
    float *input_data_c2 = input_data + image_size * 2;
    int i = 0;
    for (; i < image_size - 3 ; i += 4) {
      float32x4x3_t vin3 = vld3q_f32(image_data);
      float32x4_t vsub0 = vsubq_f32(vin3.val[0], vmean0);
      float32x4_t vsub1 = vsubq_f32(vin3.val[1], vmean1);
      float32x4_t vsub2 = vsubq_f32(vin3.val[2], vmean2);
      float32x4_t vs0 = vmulq_f32(vsub0, vscale0);
      float32x4_t vs1 = vmulq_f32(vsub1, vscale1);
      float32x4_t vs2 = vmulq_f32(vsub2, vscale2);
      vst1q_f32(input_data_c0, vs0);
      vst1q_f32(input_data_c1, vs1);
      vst1q_f32(input_data_c2, vs2);
      image_data += 12;
      input_data_c0 += 4;
      input_data_c1 += 4;
      input_data_c2 += 4;
    }
    for (; i < image_size; i++) {
      *(input_data_c0++) = (*(image_data++) - INPUT_MEAN[0]) / INPUT_STD[0];
      *(input_data_c1++) = (*(image_data++) - INPUT_MEAN[1]) / INPUT_STD[1];
      *(input_data_c2++) = (*(image_data++) - INPUT_MEAN[2]) / INPUT_STD[2];
    }

    // Detection Model Run
    predictor->Run();

    // Get Output Tensor
    std::unique_ptr<const Tensor> output_tensor(std::move(predictor->GetOutput(0)));
    auto output_data = output_tensor->data<float>();
    auto output_shape = output_tensor->shape();
    int64_t outputSize = ShapeProduction(output_shape);
    for(int i =0 ;i<outputSize ; i++) {
//      std::cout << output_data[i] << std::endl;
      std::string class_name = "Unknown";

        if(output_data[i] > 0.9f ) {
            class_name = word_labels[i];
            std::cout << "预测结果为:" << class_name << " Acc:" << output_data[i] << std::endl;
        }

    }
}

int main(int argc, char** argv) {

    std::string scenes_model = argv[1];
    const int CPU_THREAD_NUM = 4;
    const paddle::lite_api::PowerMode CPU_POWER_MODE = paddle::lite_api::PowerMode::LITE_POWER_FULL;

    // Detection
    MobileConfig config;
    config.set_threads(CPU_THREAD_NUM);
    config.set_power_mode(CPU_POWER_MODE);
    config.set_model_from_file(scenes_model);//v2.6 API

    // Create Predictor For Detction Model
    std::shared_ptr<PaddlePredictor> predictor = CreatePaddlePredictor<MobileConfig>(config);

    if (argc == 4){
      std::string img_path = argv[2];
      std::cout << argv[2] << std::endl;

      std::string label_path = argv[3];
      std::vector<std::string> word_labels = load_labels(label_path);

      cv::Mat photo = imread(img_path, cv::IMREAD_COLOR);
      RunModel(photo,word_labels,predictor);
      cv::cvtColor(photo, photo, CV_RGB2BGR);//RGB->BGR 还原正常颜色显示
      cv::imshow("scenes", photo);
      cv::waitKey(0);
    }else {
      exit(1);
    }
    return 0;
}
