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

#include "pipeline.h"
#include <iostream>

std::vector<std::string> recText;

cv::Mat GetRotateCropImage(cv::Mat srcimage,
                           std::vector<std::vector<int>> box) {
  cv::Mat image;
  srcimage.copyTo(image);
  std::vector<std::vector<int>> points = box;

  int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
  int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
  int left = int(*std::min_element(x_collect, x_collect + 4));
  int right = int(*std::max_element(x_collect, x_collect + 4));
  int top = int(*std::min_element(y_collect, y_collect + 4));
  int bottom = int(*std::max_element(y_collect, y_collect + 4));

  cv::Mat img_crop;
  image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

  for (int i = 0; i < points.size(); i++) {
    points[i][0] -= left;
    points[i][1] -= top;
  }

  int img_crop_width =
      static_cast<int>(sqrt(pow(points[0][0] - points[1][0], 2) +
                            pow(points[0][1] - points[1][1], 2)));
  int img_crop_height =
      static_cast<int>(sqrt(pow(points[0][0] - points[3][0], 2) +
                            pow(points[0][1] - points[3][1], 2)));

  cv::Point2f pts_std[4];
  pts_std[0] = cv::Point2f(0., 0.);
  pts_std[1] = cv::Point2f(img_crop_width, 0.);
  pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
  pts_std[3] = cv::Point2f(0.f, img_crop_height);

  cv::Point2f pointsf[4];
  pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
  pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
  pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
  pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

  cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

  cv::Mat dst_img;
  cv::warpPerspective(img_crop, dst_img, M,
                      cv::Size(img_crop_width, img_crop_height),
                      cv::BORDER_REPLICATE);

  const float ratio = 1.5;
  if (static_cast<float>(dst_img.rows) >=
      static_cast<float>(dst_img.cols) * ratio) {
    cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
    cv::transpose(dst_img, srcCopy);
    cv::flip(srcCopy, srcCopy, 0);
    return srcCopy;
  } else {
    return dst_img;
  }
}

std::vector<std::string> ReadDict(std::string path) {
  std::ifstream in(path);
  std::string filename;
  std::string line;
  std::vector<std::string> m_vec;
  if (in) {
    while (getline(in, line)) {
      m_vec.push_back(line);
    }
  } else {
    std::cout << "no such file" << std::endl;
  }
  return m_vec;
}

std::vector<std::string> split(const std::string &str,
                               const std::string &delim) {
  std::vector<std::string> res;
  if ("" == str)
    return res;
  char *strs = new char[str.length() + 1];
  std::strcpy(strs, str.c_str());

  char *d = new char[delim.length() + 1];
  std::strcpy(d, delim.c_str());

  char *p = std::strtok(strs, d);
  while (p) {
    std::string s = p;
    res.push_back(s);
    p = std::strtok(nullptr, d);
  }

  return res;
}

std::map<std::string, double> LoadConfigTxt(std::string config_path) {
  auto config = ReadDict(config_path);

  std::map<std::string, double> dict;
  for (int i = 0; i < config.size(); i++) {
    std::vector<std::string> res = split(config[i], " ");
    dict[res[0]] = stod(res[1]);
  }
  return dict;
}

cv::Mat Visualization(cv::Mat srcimg,
                      std::vector<std::vector<std::vector<int>>> boxes,
                      std::string output_image_path) {
  cv::Point rook_points[boxes.size()][4];
  for (int n = 0; n < boxes.size(); n++) {
    for (int m = 0; m < boxes[0].size(); m++) {
      rook_points[n][m] = cv::Point(static_cast<int>(boxes[n][m][0]),
                                    static_cast<int>(boxes[n][m][1]));
    }
  }
  cv::Mat img_vis;
  srcimg.copyTo(img_vis);
  for (int n = 0; n < boxes.size(); n++) {
    const cv::Point *ppt[1] = {rook_points[n]};
    int npt[] = {4};
    cv::polylines(img_vis, ppt, npt, 1, 1, CV_RGB(0, 255, 0), 2, 8, 0);
  }
  //cv::Mat img_vis_bgr;
  //cv::cvtColor(img_vis, img_vis_bgr, cv::COLOR_RGBA2BGR);
  //cv::imwrite(output_image_path, img_vis_bgr);
  return img_vis;
}

void Pipeline::VisualizeResults(std::vector<std::string> rec_text,
                                std::vector<float> rec_text_score,
                                cv::Mat *rgbaImage,
                                double *visualizeResultsTime) {
  auto t = GetCurrentTime();
  char text[255];
  cv::Scalar color = cv::Scalar(255, 255, 255);
  int font_face = cv::FONT_HERSHEY_PLAIN;
  double font_scale = 1.f;
  float thickness = 1;
  sprintf(text, "OCR results");
  cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
  text_size.height *= 1.25f;
  cv::Point2d offset(10, text_size.height + 15);
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);

  for (int i = 0; i < rec_text.size(); i++) {
    LOGD("debug=== line %d %s, %f", i, rec_text[i].c_str(), rec_text_score[i]);
    sprintf(text, "line: %d %s  %f", i, rec_text[i].c_str(), rec_text_score[i]);
    offset.y += text_size.height;
    cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
                thickness);
  }
  *visualizeResultsTime = GetElapsedTime(t);
  LOGD("VisualizeResults costs %f ms", *visualizeResultsTime);
}

void Pipeline::VisualizeStatus(double readGLFBOTime, double writeGLTextureTime,
                               double predictTime,
                               std::vector<std::string> rec_text,
                               std::vector<float> rec_text_score,
                               double visualizeResultsTime,
                               cv::Mat *rgbaImage) {
  char text[255];
  cv::Scalar color = cv::Scalar(255, 255, 255);
  int font_face = cv::FONT_HERSHEY_PLAIN;
  double font_scale = 1.f;
  float thickness = 1;
  sprintf(text, "Read GLFBO time: %.1f ms", readGLFBOTime);
  cv::Size text_size =
      cv::getTextSize(text, font_face, font_scale, thickness, nullptr);
  text_size.height *= 1.25f;
  cv::Point2d offset(10, text_size.height + 15);
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  sprintf(text, "Write GLTexture time: %.1f ms", writeGLTextureTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  // predict time
  sprintf(text, "OCR all process time: %.1f ms", predictTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
  // Visualize results
  sprintf(text, "Visualize results time: %.1f ms", visualizeResultsTime);
  offset.y += text_size.height;
  cv::putText(*rgbaImage, text, offset, font_face, font_scale, color,
              thickness);
}

Pipeline::Pipeline(const std::string &detModelDir,
                   const std::string &classModelDir,
                   const std::string &cPUPowerMode, const int cPUThreadNum,
                   const std::string &config_path,
                   const std::string &dict_path) {

  detPredictor_.reset(
      new DetPredictor(detModelDir, cPUThreadNum, cPUPowerMode));
  classPredictor_.reset(
      new ClassPredictor(classModelDir, cPUThreadNum, cPUPowerMode));
  Config_ = LoadConfigTxt(config_path);
    items_list_ = ReadDict(dict_path);

}

std::vector<std::string> Pipeline::getOCRResults()
{
    return recText;
}

bool Pipeline::Process_val(int inTextureId, int outTextureId, int textureWidth,
                           int textureHeight, std::string savedImagePath) {
  double readGLFBOTime = 0, writeGLTextureTime = 0;
  double visualizeResultsTime = 0, predictTime = 0;
  int height = 448;
  int width = 448;

  cv::Mat rgbaImage = cv::Mat::zeros(cv::Size(textureWidth,
                                              textureHeight),CV_8UC3);
  CreateRGBAImageFromGLFBOTexture(textureWidth, textureHeight, &rgbaImage,
                                  &readGLFBOTime);

  // change to 3-channel
  cv::Mat bgrImage;
  cv::cvtColor(rgbaImage, bgrImage, cv::COLOR_RGBA2BGR);
  cv::Mat bgrImage_resize;
  cv::resize(bgrImage, bgrImage_resize, cv::Size(width, height));

  cv::Mat srcimg;
  bgrImage_resize.copyTo(srcimg);
  // Stage1: rec
  auto t = GetCurrentTime();
  // det predict
  auto boxes =
      detPredictor_->Predict(srcimg, Config_, nullptr, nullptr, nullptr);

  std::vector<float> mean = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};

  cv::Mat img;
  bgrImage_resize.copyTo(img);
  cv::Mat crop_img;

  std::vector<std::string> rec_text;
  std::vector<float> rec_text_score;
  LOGD("debug===boxes: %d", boxes.size());

  for (int i = boxes.size() - 1; i >= 0; i--) {
    crop_img = GetRotateCropImage(img, boxes[i]);

    auto res = classPredictor_->Predict(crop_img, nullptr, nullptr, nullptr,
                                       items_list_);
    rec_text.push_back(res.first);
    //rec_text.push_back((std::string) "Item");
    rec_text_score.push_back(res.second);
    //rec_text_score.push_back(1.0f);
  }

  recText = rec_text;
  predictTime = GetElapsedTime(t);
  // visualization
  auto img_res = Visualization(bgrImage_resize, boxes, std::move(savedImagePath));
  cv::Mat img_vis;
  cv::resize(img_res, img_vis, cv::Size(textureWidth, textureHeight));
  cv::cvtColor(img_vis, img_vis, cv::COLOR_BGR2RGBA);
  // show ocr results on image
  VisualizeResults(rec_text, rec_text_score, &img_vis, &visualizeResultsTime);
  //VisualizeStatus(readGLFBOTime, writeGLTextureTime, predictTime, rec_text,
  //                rec_text_score, visualizeResultsTime, &img_vis);

  //cv::Mat testImage(textureHeight, textureWidth, CV_8UC3, cv::Scalar(0,255,0));
  //cv::cvtColor(testImage, testImage, cv::COLOR_BGR2RGBA);

   WriteRGBAImageBackToGLTexture(img_vis, outTextureId, &writeGLTextureTime);

  return true;
}