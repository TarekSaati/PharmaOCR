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

#include "Native.h"
#include "pipeline.h"
#include <android/log.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     com_baidu_paddle_lite_demo_ocr_db_crnn_Native
 * Method:    nativeInit
 * Signature:
 * (Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;Ljava/lang/String;ILjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL
Java_com_baidu_paddle_lite_demo_ppocr_1demo_Native_nativeInit(
    JNIEnv *env, jclass thiz, jstring jDetModelPath, jstring jClassModelPath,
    jstring jConfigPath, jstring jLabelPath,
    jint cpuThreadNum, jstring jCPUPowerMode) {
  std::string detModelPath = jstring_to_cpp_string(env, jDetModelPath);
  std::string classModelPath = jstring_to_cpp_string(env, jClassModelPath);
  std::string configPath = jstring_to_cpp_string(env, jConfigPath);
  std::string labelPath = jstring_to_cpp_string(env, jLabelPath);
  std::string cpuPowerMode = jstring_to_cpp_string(env, jCPUPowerMode);

  return reinterpret_cast<jlong>(
      new Pipeline(detModelPath, classModelPath, cpuPowerMode,
                   cpuThreadNum, configPath, labelPath));
}

/*
 * Class:     com_baidu_paddle_lite_demo_ocr_db_crnn_Native
 * Method:    nativeRelease
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_ppocr_1demo_Native_nativeRelease(JNIEnv *env,
                                                                 jclass thiz,
                                                                 jlong ctx) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
  delete pipeline;
  return JNI_TRUE;
}

/*
 * Class:     com_baidu_paddle_lite_demo_ocr_db_crnn_Native
 * Method:    nativeProcess
 * Signature: (JIIIILjava/lang/String;)Z
 */
JNIEXPORT jboolean JNICALL
Java_com_baidu_paddle_lite_demo_ppocr_1demo_Native_nativeProcess(
    JNIEnv *env, jclass thiz, jlong ctx, jint inTextureId, jint outTextureId,
    jint textureWidth, jint textureHeight, jstring jsavedImagePath) {
  if (ctx == 0) {
    return JNI_FALSE;
  }
  std::string savedImagePath = jstring_to_cpp_string(env, jsavedImagePath);
  Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
  return pipeline->Process_val(inTextureId, outTextureId, textureWidth,
                               textureHeight, savedImagePath);
}

JNIEXPORT jobjectArray JNICALL
Java_com_baidu_paddle_lite_demo_ppocr_1demo_Native_nativeOCRResults(
        JNIEnv *env, jclass thiz, jlong ctx) {
    if (ctx == 0) {
        return JNI_FALSE;
    }

    Pipeline *pipeline = reinterpret_cast<Pipeline *>(ctx);
    std::vector<std::string> data = pipeline->getOCRResults();
    if (data.empty())
        return nullptr;
    jobjectArray ret;
    ret= (jobjectArray)env->NewObjectArray(data.size(),env->FindClass("java/lang/String"),env->NewStringUTF(""));

    for(int i=0; i<data.size(); i++) env->SetObjectArrayElement(ret,i,env->NewStringUTF(
                (const char *) data[i].c_str()));

    return(ret);
}

#ifdef __cplusplus
}
#endif
