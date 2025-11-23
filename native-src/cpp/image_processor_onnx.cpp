//
// Created by ghima on 21-11-2025.
//
#include "image_processor_onnx.h"
#include "onnxruntime_cxx_api.h"
#include "android/log.h"
#include "android/bitmap.h"
#include "image_processor_tensorflow.h"

#define LOG_TAG "core_native_image"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml {
    Ort::Session *ImageProcessorOnnx::session = nullptr;
    Ort::Env ImageProcessorOnnx::envOrt = Ort::Env{ORT_LOGGING_LEVEL_WARNING, "nsfw"};

    void ImageProcessorOnnx::get_pixels_and_process_image(JNIEnv *env, jobject bitmap,
                                                          const char *modelName) {
        void *pixels;
        AndroidBitmapInfo info{};
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOG_ERROR("Failed to get the bitmap info ");
            return;
        }
        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_ERROR("Invalid format for bit map ");
            return;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock the bit map pixels");
            return;
        }
        uint8_t *scaledImage = new uint8_t[640 * 640 * 3];
        float *tensorPlanar = new float[640 * 640 * 3];
        resize_image_bilinear_letter(reinterpret_cast<uint8_t *>(pixels), info.width, info.height,
                                     info.stride,
                                     scaledImage, 640, 640);
        ImageProcessorNsfwLite::prepare_tensor_planar(tensorPlanar, scaledImage, 640, 640);
        process_image_with_onnx_model(modelName, tensorPlanar);
        delete[] scaledImage;
        delete[] tensorPlanar;
    }

    void
    ImageProcessorOnnx::scale_image_nearest(uint8_t *src, int sw, int sh, int stride, uint8_t *dst,
                                            int dw,
                                            int dh) {
        float xRatio = sw / (float) dw;
        float yRatio = sh / (float) dh;

        for (int y = 0; y < dh; y++) {
            int sy = (int) (y * yRatio);
            for (int x = 0; x < dw; x++) {
                int sx = (int) (x * xRatio);
                uint8_t *src_pixel = src + (sy * sw + sx) * 4;
                uint8_t *dst_pixel = dst + (y * dw + x) * 4;

                dst_pixel[0] = src_pixel[0];
                dst_pixel[1] = src_pixel[1];
                dst_pixel[2] = src_pixel[2];
                dst_pixel[3] = src_pixel[3];
            }
        }
    }

    void ImageProcessorOnnx::resize_image_bilinear_letter(const uint8_t *src, int sw, int sh,
                                                          int sStride,
                                                          uint8_t *dst,
                                                          int dw, int dh) {
        const int padColor = 114;
        for (int i = 0; i < dh; i++) {
            for (int j = 0; j < dw; j++) {
                uint8_t *p = dst + (i * dw + j) * 3;
                p[0] = p[1] = p[2] = padColor;
            }
        }
        float scale = std::min(dw / (float) sw, dh / (float) sh);
        int new_dh = (int) (sh * scale);
        int new_dw = (int) (sw * scale);

        int pad_left = (dw - new_dw) / 2;
        int pad_top = (dh - new_dh) / 2;


        for (int y = 0; y < new_dh; y++) {
            float yM = (y + .5) / scale - .5f;
            // float yM = (y) / scale;
            float y0 = std::floorf(yM);
            float y1 = y0 + 1;
            if (y0 < 0) y0 = 0;
            if (y1 >= sh) y1 = sh - 1;
            float wy1 = yM - y0;
            float wy0 = 1 - wy1;

            for (int x = 0; x < new_dw; x++) {
                float xM = (x + .5f) / scale - .5f;
                //float xM = x / scale;
                float x0 = std::floorf(xM);
                float x1 = x0 + 1;

                if (x0 < 0) x0 = 0;
                if (x1 >= sw) x1 = sw - 1;

                float wx1 = xM - x0;
                float wx0 = 1 - wx1;
                const uint8_t *src_x0y0 = src + (int) ((y0 * sStride) + x0 * 4);
                const uint8_t *src_x0y1 = src + (int) ((y1 * sStride) + x0 * 4);
                const uint8_t *src_x1y0 = src + (int) ((y0 * sStride) + x1 * 4);
                const uint8_t *src_x1y1 = src + (int) ((y1 * sStride) + x1 * 4);

                // Getting RGB channels
                for (int c = 0; c < 4; c++) {
                    float v0 = (src_x0y0[c] * wx0 + src_x1y0[c] * wx1) * wy0;
                    float v1 = (src_x0y1[c] * wx0 + src_x1y1[c] * wx1) * wy1;

                    dst[((y + pad_top) * dw + x + pad_left) * 4 + c] = (uint8_t) (v0 + v1);
                }

            }
        }
    }

    void ImageProcessorOnnx::init_model(const char *modelPath, std::string &inputName,
                                        std::string &outputName) {
        Ort::SessionOptions options;
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session = new Ort::Session(envOrt, modelPath, options);
        Ort::AllocatorWithDefaultOptions allocator;
        auto inputNamedAllocator = session->GetInputNameAllocated(0, allocator);
        Ort::TypeInfo inputInfo = session->GetInputTypeInfo(0);
        auto inputTensorInfo = inputInfo.GetTensorTypeAndShapeInfo();
        std::vector<int64_t> inputDims = inputTensorInfo.GetShape();

        LOG_INFO("Model Input Name %s with dims %ld, %ld, %ld, %ld and type is %d",
                 inputNamedAllocator.get(),
                 inputDims[0], inputDims[1], inputDims[2], inputDims[3],
                 inputTensorInfo.GetElementType());

        auto outNameAllocator = session->GetOutputNameAllocated(0, allocator);
        inputName = inputNamedAllocator.get();
        LOG_INFO("Model Output Name %s", outNameAllocator.get());
        Ort::TypeInfo outputInfo = session->GetOutputTypeInfo(0);
        auto outputTensorInfo = outputInfo.GetTensorTypeAndShapeInfo();
        outputName = outNameAllocator.get();
        LOG_INFO("Model Out Info %ld, %ld, %ld", outputTensorInfo.GetShape()[0],
                 outputTensorInfo.GetShape()[1], outputTensorInfo.GetShape()[2]);

    }

    void ImageProcessorOnnx::process_image_with_onnx_model(const char *modelName, float *inputVal) {

        std::string inputName;
        std::string outputName;
        init_model(modelName, inputName, outputName);

        std::array<int64_t, 4> shape{1, 3, 640, 640};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                                OrtMemType::OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
                memoryInfo,
                inputVal,
                3 * 640 * 640,
                shape.data(),
                shape.size()
        );
        const char *inputNames[] = {inputName.c_str()};
        const char *outputNames[] = {outputName.c_str()};

        LOG_INFO("NOW RUNNING THE MODEL %s %s", inputName.c_str(), outputName.c_str());
        auto output = session->Run(Ort::RunOptions{nullptr},
                                   inputNames,
                                   &inputTensor, 1,
                                   outputNames, 1);
        auto outputShape = output[0].GetTensorTypeAndShapeInfo();
        auto dims = outputShape.GetShape();
        LOG_INFO("Runtime output shape:");
        for (int i = 0; i < dims.size(); i++) {
            LOG_INFO("dim[%d] = %ld", i, dims[i]);
        }
        int channels = outputShape.GetShape()[1];
        int predictions = outputShape.GetShape()[2];

        float *out = output[0].GetTensorMutableData<float>();
        for (int k = 0; k < predictions; k++) {

            float cx = out[0 * predictions + k];
            float cy = out[1 * predictions + k];
            float w = out[2 * predictions + k];
            float h = out[3 * predictions + k];
            float obj = sigmoid(out[4 * predictions + k]);

            for (int c = 5; c < channels; c++) {
                float cls = sigmoid(out[c * predictions + k]);
                float score = obj * cls;
                LOG_INFO("class %d score %f", c - 5, score);
            }
        }

    }
}