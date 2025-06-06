
#include "class_process.h"

#include <utility>
#include "utils.h"

const std::vector<int> class_image_shape{1, 32, 64};

cv::Mat ResizeImg(const cv::Mat& img) {

  cv::Mat resize_img;
  cv::resize(img, resize_img,
             cv::Size(class_image_shape[2], class_image_shape[1]),
             0.f, 0.f, cv::INTER_LINEAR);

  return resize_img;
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

  cv::Mat resize_img = ResizeImg(srcimg);
  cv::cvtColor(resize_img, resize_img, cv::COLOR_BGR2GRAY);
  resize_img.convertTo(resize_img, CV_32FC1, 1 / 255.f);

  const float *dimg = reinterpret_cast<const float *>(resize_img.data);

  std::unique_ptr<Tensor> input_tensor0(std::move(predictor_->GetInput(0)));
  input_tensor0->Resize({1, 1, resize_img.rows, resize_img.cols});
  auto *data0 = input_tensor0->mutable_data<float>();
  const float gray_mean[] = {0.5f};
  const float gray_std[] = {0.5f};
  NHWC1ToNC1HW(dimg, data0, gray_mean, gray_std,
                 resize_img.cols, resize_img.rows);
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

  for (int id = 0; id < items_list.size(); id++) {

      if (predict_batch[id] > max_score) {
          max_score = predict_batch[id];
          max_index = id;
      }
  }
  /*
    max_index = int(Argmax(&predict_batch[0], &predict_batch[predict_shape[1]]));
    max_score =
            float(*std::max_element(&predict_batch[0], &predict_batch[predict_shape[1]]));
            */
  //str_res = items_list.size() > max_index ? items_list[max_index] : "Unknown";
  str_res = items_list[max_index];
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
