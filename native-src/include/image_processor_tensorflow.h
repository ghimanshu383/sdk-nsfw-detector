//
// Created by ghima on 19-11-2025.
//

#ifndef OSFEATURENDKDEMO_IMAGE_PROCESSOR_TENSORFLOW_H
#define OSFEATURENDKDEMO_IMAGE_PROCESSOR_TENSORFLOW_H

#include <cstdint>
#include "jni.h"
#include <vector>
#include <mutex>
#include "tensorflow/lite/c/c_api.h"
#include "tensorflow/lite/c/c_api_experimental.h"

namespace ml {
    struct NsfwAggregation {
        float drawing = 0.0;
        float hentai = 0.0f;
        float neutral = 0.0;
        float porn = 0.0f;
        float sexy = 0.0f;
    };

    class ImageProcessorNsfwLite {
        static std::mutex mutex_;
        static TfLiteModel *model;
        static TfLiteInterpreterOptions *interpreterOptions;

        static void init_model(const char *modelPath);

        static void clean_up_model();

        static bool process_model(void *imageTensor,
                                  size_t tensorSize, const char *modelPath,
                                  NsfwAggregation &nsfwAggregation);

        static void update_nsfw_score(float *score, NsfwAggregation &nsfwAggregation);


        static void
        extract_window_rgba(uint8_t *src, int strideBytes, int startX, int startY, int ww, int wh,
                            uint8_t *dst);


    public:
        static jobject get_nsfw_factors_for_java_bridge(JNIEnv *env, NsfwAggregation &aggregation);

        static void prepare_tensor_planar(float *out, uint8_t *rgba, int w, int h);

        static void
        prepare_tensor_planar_neon(float *out, uint8_t *rgba, int w, int h);

        static void prepare_tensor_non_planar(float *out, uint8_t *rgb, int w, int h);

        static void prepare_tensor_non_planar_neon(float *out, uint8_t *rgba, int w, int h);

        static bool
        get_pixels_and_process_image_sliding_window(JNIEnv *env, jobject bitmap,
                                                    const char *modelPath,
                                                    NsfwAggregation& aggregation);

        static bool
        get_pixels_and_process_image_single_pass(JNIEnv *env, jobject bitmap, const char *modelPath,
                                                 NsfwAggregation &nsfwAggregation);


    };
}
#endif //OSFEATURENDKDEMO_IMAGE_PROCESSOR_TENSORFLOW_H
