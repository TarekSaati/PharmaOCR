
#include "class_process.h"

#include <utility>
#include "utils.h"
#include <vector>
#include <cmath>
#include <numeric>

const std::vector<int> class_image_shape{1, 64, 128};

cv::Mat ResizeImg(const cv::Mat& img) {

  cv::Mat resize_img;
  cv::resize(img, resize_img,
             cv::Size(class_image_shape[2], class_image_shape[1]),
             0.f, 0.f, cv::INTER_LINEAR);

  return resize_img;
}

std::vector<float> softmax(const std::vector<float>& input) {
    std::vector<float> output(input.size());
    float max_val = *std::max_element(input.begin(), input.end());

    float sum_exp = 0.0;
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum_exp += output[i];
    }

    for (float & i : output) {
        i /= sum_exp;
    }
    return output;
}

template <class ForwardIterator>
inline size_t Argmax(ForwardIterator first, ForwardIterator last) {
  return std::distance(first, std::max_element(first, last));
}

ClassPredictor::ClassPredictor(const std::string &modelDir, const int cpuThreadNum,
                           const std::string &cpuPowerMode) {
  paddle::lite_api::MobileConfig config;
  config.set_model_from_file(modelDir);
  config.set_threads(cpuThreadNum);
  config.set_power_mode(ParsePowerMode(cpuPowerMode));
  predictor_ =
      paddle::lite_api::CreatePaddlePredictor<paddle::lite_api::MobileConfig>(
          config);
}

void ClassPredictor::Preprocess(const cv::Mat &srcimg) {

    cv::Mat img = srcimg;
  cv::cvtColor(img, img, cv::COLOR_BGR2GRAY);
  cv::equalizeHist(img, img);
    img = 255 - img;
  int thresh_val = 160;
  int max_val = 255;
  cv::threshold(img, img, thresh_val, max_val, cv::THRESH_BINARY);

  int erosion_size = 0;
  cv::Mat element = cv::getStructuringElement( cv::MORPH_RECT, cv::Size( 2*erosion_size + 1, 2*erosion_size+1 ),
                                         cv::Point( erosion_size, erosion_size ) );
    cv::erode(img, img, element);
    cv::dilate(img, img, element);
    img.convertTo(img, CV_32FC1, 1/255.0);
    cv::Mat resize_img = ResizeImg(img);
  const float *dimg = reinterpret_cast<const float *>(resize_img.data);

  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
  input_tensor0->Resize({1, 1, resize_img.rows, resize_img.cols});
  auto *data0 = input_tensor0->mutable_data<float>();
    const float mean[] = {0.5f};
    const float scale[] = {2.f};
    NHWC1ToNC1HW(dimg, data0, mean, scale, resize_img.cols, resize_img.rows);
}

std::pair<std::string, float>
ClassPredictor::Postprocess(const cv::Mat &rgbaImage,
                          std::vector<std::string> items_list) {
  // Get output and run postprocess
  std::unique_ptr<const Tensor> output_tensor0(
      std::move(predictor_->GetOutput(0)));
  auto *predict_batch = output_tensor0->data<float>();
  auto predict_shape = output_tensor0->shape();
    LOGD("ocr cpp output tensor[%d] size {%d, %d}", 0, predict_shape[0], predict_shape[1]);

    // ctc decode
  std::string str_res;
  int max_index;
  float max_score = -std::numeric_limits<float>::max();
  std::vector<float> logits {predict_batch, predict_batch + items_list.size()};
  std::vector<float> pred_probabs = softmax(logits);
    LOGD("ocr cpp output tensor[%d] ptr/value {%f, %f, %f, %f}", logits.size(), logits[0], logits[1]
    , logits[2], logits[3]);
    LOGD("softmax values {%f, %f, %f, %f}", pred_probabs.size(), pred_probabs[0], pred_probabs[1]
    , pred_probabs[2], pred_probabs[3]);
    std::string itemsList[] = {"Toplexil", "Flagyl", "Doprane", "Unknown"};
  for (int id = 0; id < 4; id++) {

      if (pred_probabs[id] > max_score) {
          max_score = pred_probabs[id];
          max_index = id;
      }
  }
/*
    max_index = int(Argmax(&predict_batch[0], &predict_batch[predict_shape[1]]));
    max_score =
            float(*std::max_element(&predict_batch[0], &predict_batch[predict_shape[1]]));
    //str_res = items_list.size() > max_index ? items_list[max_index] : "Unknown";
    */
  str_res = itemsList[max_index];
  return std::make_pair(str_res, max_score);
}

std::pair<std::string, float>
ClassPredictor::Predict(const cv::Mat &rgbaImage, double *preprocessTime,
                      double *predictTime, double *postprocessTime,
                      std::vector<std::string> items_list) {
  // auto t = GetCurrentTime();
  Preprocess(rgbaImage);
  // *preprocessTime = GetElapsedTime(t);

  // t = GetCurrentTime();
  predictor_->Run();
  // *predictTime = GetElapsedTime(t);

  // t = GetCurrentTime();
  auto res = Postprocess(rgbaImage, std::move(items_list));
  // *postprocessTime = GetElapsedTime(t);
  return res;
}
