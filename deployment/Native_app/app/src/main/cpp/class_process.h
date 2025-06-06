
#pragma once

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "paddle_api.h"
#include "utils.h"
using namespace paddle::lite_api; // NOLINT

class ClassPredictor {
public:
  explicit ClassPredictor(const std::string &modelDir, const int cpuThreadNum,
                        const std::string &cpuPowerMode);

  std::pair<std::string, float>
  Predict(const cv::Mat &rgbaImage, double *preprocessTime, double *predictTime,
          double *postprocessTime, std::vector<std::string> items_list);

private:
  void Preprocess(const cv::Mat &rgbaImage);
  std::pair<std::string, float>
  Postprocess(const cv::Mat &rgbaImage,
              std::vector<std::string> items_list);

private:
  std::shared_ptr<paddle::lite_api::PaddlePredictor> predictor_;
};