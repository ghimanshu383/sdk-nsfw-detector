//
// Created by ghima on 21-11-2025.
//

#ifndef OSFEATURENDKDEMO_IMAGE_PROCESSOR_ONNX_H
#define OSFEATURENDKDEMO_IMAGE_PROCESSOR_ONNX_H

#include "jni.h"
#include "onnxruntime_cxx_api.h"
#include <string>

namespace ml {
    class ImageProcessorOnnx {
        static Ort::Env envOrt;
        static Ort::Session *session;
    public:
        static void
        get_pixels_and_process_image(JNIEnv *env, jobject bitmap, const char *modelName);

        static void
        resize_image_bilinear_letter(const uint8_t *src, int sw, int sh, int sStride, uint8_t *dst,
                                     int dw, int dh);
        static void scale_image_nearest(uint8_t *src, int sw, int sh, int stride, uint8_t *dst, int dw,
                                        int dh);

    private:


        static void process_image_with_onnx_model(const char *modelName, float *inputTensor);


        static void
        init_model(const char *modelPath, std::string &inputName, std::string &outputName);

        static float sigmoid(float x) {
            if (x >= 0) {
                float z = expf(-x);
                return 1.f / (1.f + z);
            } else {
                float z = expf(x);
                return z / (1.f + z);
            }
        }
    };
}
#endif //OSFEATURENDKDEMO_IMAGE_PROCESSOR_ONNX_H
