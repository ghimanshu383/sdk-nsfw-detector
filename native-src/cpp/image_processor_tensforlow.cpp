//
// Created by ghima on 19-11-2025.
//
#include <string>
#include <vector>
#include "image_processor_tensorflow.h"
#include "android/bitmap.h"
#include "android/log.h"
#include "arm_neon.h"
#include "image_processor_onnx.h"
#include "ThreadPool.h"

#define LOG_TAG "core_native_image"
#define LOG_INFO(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOG_ERROR(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace ml {
    TfLiteModel *ImageProcessorNsfwLite::model = nullptr;
    TfLiteInterpreterOptions *ImageProcessorNsfwLite::interpreterOptions;
    std::mutex ImageProcessorNsfwLite::mutex_ = {};

    bool
    ImageProcessorNsfwLite::get_pixels_and_process_image_sliding_window(JNIEnv *env, jobject bitmap,
                                                                        const char *modelPath,
                                                                        NsfwAggregation &nsfwAggregation) {
        AndroidBitmapInfo info;
        void *pixels;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOG_INFO("Failed to get bitmap info");
            return false;
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_INFO("Invalid bit map format required RGBA 8888");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock android pixels");
            return false;
        }

        // preparing the tensor with the resized image;
//        uint8_t *resized_image = new uint8_t[224 * 224 * 3];
//        ImageProcessorOnnx::resize_image_bilinear_letter(reinterpret_cast<uint8_t *>(pixels),
//                                                         info.width, info.height, info.stride,
//                                                         resized_image, 224, 224);
//        prepare_tensor_non_planar(tensor, resized_image, 224, 224);

        const int windowSize = 224;
        const int stride = 112;

        init_model(modelPath);
        size_t numThreads = std::thread::hardware_concurrency();
        ThreadPool threadPool{numThreads};

        int group = (info.height - windowSize - 1) / windowSize;
        int per = group / numThreads;
        LOG_INFO(" PER %d NUM threads %zu", per, numThreads);
        int rem = group % numThreads;
        int gStart = 0;
        for (int t = 0; t < numThreads; t++) {
            int gCount = per + (t < rem ? 1 : 0);
            int startY = gStart;
            int endY = startY + gCount;
            gStart = endY;

            threadPool.enqueue_task([&, startY, endY]() -> void {
                uint8_t *window = new uint8_t[windowSize * windowSize * 4];
                float *tensor = new float[224 * 224 * 3];
                for (int wy = startY * windowSize; wy <= windowSize * endY; wy += stride) {
                    for (int wx = 0; wx + windowSize <= info.width; wx += stride) {
                        extract_window_rgba(reinterpret_cast<uint8_t *>(pixels), info.stride,
                                            wx, wy,
                                            windowSize, windowSize, window);
                        prepare_tensor_non_planar_neon(tensor, window, 224, 224);
                        process_model(tensor, sizeof(float) * 224 * 224 * 3, modelPath,
                                      nsfwAggregation);
                    }
                }
                delete[] window;
                delete[] tensor;
            });
        }
        threadPool.join_all();
        clean_up_model();
        LOG_INFO("Drawing %f", nsfwAggregation.drawing);
        LOG_INFO("Hentai %f", nsfwAggregation.hentai);
        LOG_INFO("Neutral %f", nsfwAggregation.neutral);
        LOG_INFO("Porn %f", nsfwAggregation.porn);
        LOG_INFO("Sexy %f", nsfwAggregation.sexy);
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    }

    void
    ImageProcessorNsfwLite::extract_window_rgba(uint8_t *src, int strideBytes, int startX,
                                                int startY,
                                                int ww, int wh, uint8_t *dst) {
        const int bytesPerPixel = 4;

        for (int y = 0; y < wh; ++y) {
            uint8_t *rowPtr = src + (size_t) (startY + y) * (size_t) strideBytes;
            uint8_t *srcIndex = rowPtr + (size_t) startX * bytesPerPixel;
            uint8_t *dstIndex = dst + (size_t) y * ww * bytesPerPixel;
            memcpy(dstIndex, srcIndex, (size_t) ww * bytesPerPixel);
        }
    }

    void ImageProcessorNsfwLite::prepare_tensor_non_planar(float *out, uint8_t *rgb, int w, int h) {
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int srcIndex = (y * w + x) * 4;
                int dstIndex = (y * w + x) * 3;
                out[dstIndex] = rgb[srcIndex] / 255.0f;
                out[dstIndex + 1] = rgb[srcIndex + 1] / 255.0f;
                out[dstIndex + 2] = rgb[srcIndex + 2] / 255.0f;
            }
        }
    }

    void
    ImageProcessorNsfwLite::prepare_tensor_planar_neon(float *out, uint8_t *rgba, int w, int h) {
        int HW = w * h;
        float32x4_t inv255 = vdupq_n_f32(1.0f / 255.0f);
        int rem = w % 16;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x += 16) {
                uint8x16x4_t ch = vld4q_u8(rgba + (y * w + x) * 4);
                uint16x8_t r_lo = vmovl_u8(vget_low_u8(ch.val[0]));
                uint16x8_t r_hi = vmovl_u8(vget_high_u8(ch.val[0]));
                uint32x4_t r_1 = vmovl_u16(vget_low_u16(r_lo));
                uint32x4_t r_2 = vmovl_u16(vget_high_u16(r_lo));
                uint32x4_t r_3 = vmovl_u16(vget_low_u16(r_hi));
                uint32x4_t r_4 = vmovl_u16(vget_high_u16(r_hi));

                float32x4_t r_l_1 = vcvtq_f32_u32(r_1);
                float32x4_t r_l_2 = vcvtq_f32_u32(r_2);
                float32x4_t r_h_1 = vcvtq_f32_u32(r_3);
                float32x4_t r_h_2 = vcvtq_f32_u32(r_4);

                r_l_1 = vmulq_f32(r_l_1, inv255);
                r_l_2 = vmulq_f32(r_l_2, inv255);
                r_h_1 = vmulq_f32(r_h_1, inv255);
                r_h_2 = vmulq_f32(r_h_2, inv255);


                uint16x8_t g_lo = vmovl_u8(vget_low_u8(ch.val[1]));
                uint16x8_t g_hi = vmovl_u8(vget_high_u8(ch.val[1]));
                uint32x4_t g_1 = vmovl_u16(vget_low_u16(g_lo));
                uint32x4_t g_2 = vmovl_u16(vget_high_u16(g_lo));
                uint32x4_t g_3 = vmovl_u16(vget_low_u16(g_hi));
                uint32x4_t g_4 = vmovl_u16(vget_high_u16(g_hi));

                float32x4_t g_l_1 = vcvtq_f32_u32(g_1);
                float32x4_t g_l_2 = vcvtq_f32_u32(g_2);
                float32x4_t g_h_1 = vcvtq_f32_u32(g_3);
                float32x4_t g_h_2 = vcvtq_f32_u32(g_4);

                g_l_1 = vmulq_f32(g_l_1, inv255);
                g_l_2 = vmulq_f32(g_l_2, inv255);
                g_h_1 = vmulq_f32(g_h_1, inv255);
                g_h_2 = vmulq_f32(g_h_2, inv255);


                uint16x8_t b_lo = vmovl_u8(vget_low_u8(ch.val[2]));
                uint16x8_t b_hi = vmovl_u8(vget_high_u8(ch.val[2]));
                uint32x4_t b_1 = vmovl_u16(vget_low_u16(b_lo));
                uint32x4_t b_2 = vmovl_u16(vget_high_u16(b_lo));
                uint32x4_t b_3 = vmovl_u16(vget_low_u16(b_hi));
                uint32x4_t b_4 = vmovl_u16(vget_high_u16(b_hi));

                float32x4_t b_l_1 = vcvtq_f32_u32(b_1);
                float32x4_t b_l_2 = vcvtq_f32_u32(b_2);
                float32x4_t b_h_1 = vcvtq_f32_u32(b_3);
                float32x4_t b_h_2 = vcvtq_f32_u32(b_4);

                b_l_1 = vmulq_f32(b_l_1, inv255);
                b_l_2 = vmulq_f32(b_l_2, inv255);
                b_h_1 = vmulq_f32(b_h_1, inv255);
                b_h_2 = vmulq_f32(b_h_2, inv255);


                int idx = y * w + x;
                vst1q_f32(out + idx + 0, r_l_1);
                vst1q_f32(out + idx + 4, r_l_2);
                vst1q_f32(out + idx + 8, r_h_1);
                vst1q_f32(out + idx + 12, r_h_2);

                vst1q_f32(out + idx + HW + 0, g_l_1);
                vst1q_f32(out + idx + HW + 4, g_l_2);
                vst1q_f32(out + idx + HW + 8, g_h_1);
                vst1q_f32(out + idx + HW + 12, g_h_2);

                vst1q_f32(out + idx + 2 * HW + 0, b_l_1);
                vst1q_f32(out + idx + 2 * HW + 4, b_l_2);
                vst1q_f32(out + idx + 2 * HW + 8, b_h_1);
                vst1q_f32(out + idx + 2 * HW + 12, b_h_2);

            }
            // Falling back to scalar for the remaining 16 pixels if any.
            for (int x = w - rem; x < w; x++) {
                int srcIndex = (y * w + x) * 4;
                int idx = y * w + x;
                out[idx] = rgba[srcIndex + 0] / 255.0f;
                out[idx + HW] = rgba[srcIndex + 1] / 255.0f;
                out[idx + 2 * HW] = rgba[srcIndex + 2] / 255.0f;
            }
        }
    }

    void ImageProcessorNsfwLite::prepare_tensor_planar(float *out, uint8_t *rgb, int w, int h) {
        int HW = w * h;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int src = (y * w + x) * 4;
                int idx = y * w + x;

                out[idx] = rgb[src + 0] / 255.0f;
                out[idx + HW] = rgb[src + 1] / 255.0f;
                out[idx + 2 * HW] = rgb[src + 2] / 255.0f;
            }
        }
    }

    void ImageProcessorNsfwLite::prepare_tensor_non_planar_neon(float *tensor, uint8_t *rgba, int w,
                                                                int h) {
        float32x4_t inverse_255 = vdupq_n_f32(1 / 255.0f);
        int remX = w % 16;
        int limit = w - remX;
        for (int y = 0; y < h; y++) {

            for (int x = 0; x < limit; x += 16) {
                uint8x16x4_t ch = vld4q_u8(rgba + (y * w + x) * 4);
                uint16x8_t r_low = vmovl_u8(vget_low_u8(ch.val[0]));
                uint16x8_t r_high = vmovl_u8(vget_high_u8(ch.val[0]));
                uint32x4_t r_low_1 = vmovl_u16(vget_low_u16(r_low));
                uint32x4_t r_low_2 = vmovl_u16(vget_high_u16(r_low));
                uint32x4_t r_high_1 = vmovl_u16(vget_low_u16(r_high));
                uint32x4_t r_high_2 = vmovl_u16(vget_high_u16(r_high));

                float32x4_t r_l_1 = vcvtq_f32_u32(r_low_1);
                float32x4_t r_l_2 = vcvtq_f32_u32(r_low_2);
                float32x4_t r_h_1 = vcvtq_f32_u32(r_high_1);
                float32x4_t r_h_2 = vcvtq_f32_u32(r_high_2);

                r_l_1 = vmulq_f32(r_l_1, inverse_255);
                r_l_2 = vmulq_f32(r_l_2, inverse_255);
                r_h_1 = vmulq_f32(r_h_1, inverse_255);
                r_h_2 = vmulq_f32(r_h_2, inverse_255);

                uint16x8_t g_low = vmovl_u8(vget_low_u8(ch.val[1]));
                uint16x8_t g_high = vmovl_u8(vget_high_u8(ch.val[1]));
                uint32x4_t g_low_1 = vmovl_u16(vget_low_u16(g_low));
                uint32x4_t g_low_2 = vmovl_u16(vget_high_u16(g_low));
                uint32x4_t g_high_1 = vmovl_u16(vget_low_u16(g_high));
                uint32x4_t g_high_2 = vmovl_u16(vget_high_u16(g_high));

                float32x4_t g_l_1 = vcvtq_f32_u32(g_low_1);
                float32x4_t g_l_2 = vcvtq_f32_u32(g_low_2);
                float32x4_t g_h_1 = vcvtq_f32_u32(g_high_1);
                float32x4_t g_h_2 = vcvtq_f32_u32(g_high_2);

                g_l_1 = vmulq_f32(g_l_1, inverse_255);
                g_l_2 = vmulq_f32(g_l_2, inverse_255);
                g_h_1 = vmulq_f32(g_h_1, inverse_255);
                g_h_2 = vmulq_f32(g_h_2, inverse_255);


                uint16x8_t b_low = vmovl_u8(vget_low_u8(ch.val[2]));
                uint16x8_t b_high = vmovl_u8(vget_high_u8(ch.val[2]));
                uint32x4_t b_low_1 = vmovl_u16(vget_low_u16(b_low));
                uint32x4_t b_low_2 = vmovl_u16(vget_high_u16(b_low));
                uint32x4_t b_high_1 = vmovl_u16(vget_low_u16(b_high));
                uint32x4_t b_high_2 = vmovl_u16(vget_high_u16(b_high));

                float32x4_t b_l_1 = vcvtq_f32_u32(b_low_1);
                float32x4_t b_l_2 = vcvtq_f32_u32(b_low_2);
                float32x4_t b_h_1 = vcvtq_f32_u32(b_high_1);
                float32x4_t b_h_2 = vcvtq_f32_u32(b_high_2);

                b_l_1 = vmulq_f32(b_l_1, inverse_255);
                b_l_2 = vmulq_f32(b_l_2, inverse_255);
                b_h_1 = vmulq_f32(b_h_1, inverse_255);
                b_h_2 = vmulq_f32(b_h_2, inverse_255);


                float32x4x3_t out_1;
                out_1.val[0] = r_l_1;
                out_1.val[1] = g_l_1;
                out_1.val[2] = b_l_1;
                vst3q_f32(tensor + (y * w + x) * 3, out_1);

                float32x4x3_t out_2;
                out_2.val[0] = r_l_2;
                out_2.val[1] = g_l_2;
                out_2.val[2] = b_l_2;
                vst3q_f32(tensor + (y * w + (x + 4)) * 3, out_2);

                float32x4x3_t out_3;
                out_3.val[0] = r_h_1;
                out_3.val[1] = g_h_1;
                out_3.val[2] = b_h_1;
                vst3q_f32(tensor + (y * w + (x + 8)) * 3, out_3);

                float32x4x3_t out_4;
                out_4.val[0] = r_h_2;
                out_4.val[1] = g_h_2;
                out_4.val[2] = b_h_2;
                vst3q_f32(tensor + (y * w + (x + 12)) * 3, out_4);
            }
            // fallback for scalar
            for (int x = limit; x < w; x++) {
                uint8_t *srcIndex = rgba + (y * w + x) * 4;
                float *dstIndex = tensor + (y * w + x) * 3;

                dstIndex[0] = srcIndex[0] / 255.0f;
                dstIndex[1] = srcIndex[1] / 255.0f;
                dstIndex[2] = srcIndex[2] / 255.0f;
            }
        }
    }

    void ImageProcessorNsfwLite::init_model(const char *modelPath) {
        model = TfLiteModelCreateFromFile(modelPath);
        if (!model) {
            LOG_ERROR("Failed to load the model from the file %s", modelPath);
            return;
        }
        interpreterOptions = TfLiteInterpreterOptionsCreate();
        TfLiteInterpreterOptionsSetNumThreads(interpreterOptions, 2);
    }

    bool ImageProcessorNsfwLite::process_model(void *imageTensor,
                                               size_t tensorSize, const char *modelPath,
                                               NsfwAggregation &nsfwAggregation) {
        TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, interpreterOptions);

        if (!interpreter) {
            LOG_ERROR("Failed to set the interpreter for this model %s", modelPath);
            //  TfLiteInterpreterOptionsDelete(interpreterOptions);
            //TfLiteModelDelete(model);
            return false;
        }

        if (TfLiteInterpreterAllocateTensors(interpreter) != kTfLiteOk) {
            LOG_ERROR("Failed to allocate the tensor for this model");
            TfLiteInterpreterDelete(interpreter);
            // TfLiteInterpreterOptionsDelete(interpreterOptions);
            //TfLiteModelDelete(model);
            return false;
        }
        // Getting the input info
        TfLiteTensor *input = TfLiteInterpreterGetInputTensor(interpreter, 0);
        int dims = TfLiteTensorNumDims(input);
        TfLiteType inType = TfLiteTensorType(input);
        size_t byteSize = TfLiteTensorByteSize(input);
        LOG_INFO("Tensor Input Type %d", inType);
        LOG_INFO("Tensor Input Size %ld", byteSize);
        if (byteSize != tensorSize) {
            LOG_INFO("Tensor Input size and required byte size mismatch.. contact Himanshu");
            return false;
            TfLiteInterpreterDelete(interpreter);
        }

        std::string shapeString{};
        for (int i = 0; i < dims; i++) {
            int s = TfLiteTensorDim(input, i);
            if (i) shapeString += ", ";
            shapeString += std::to_string(s);
        }
        LOG_INFO("Tensor Shape [%s]", shapeString.c_str());

        const TfLiteTensor *outputTensor = TfLiteInterpreterGetOutputTensor(interpreter, 0);


        TfLiteStatus copyStatus = TfLiteTensorCopyFromBuffer(input, imageTensor, tensorSize);
        if (copyStatus != kTfLiteOk) {
            LOG_ERROR("Failed to load the tf lite input Buffer");
            TfLiteInterpreterDelete(interpreter);
            return false;
        }

        //Invoking the model
        if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk) {
            LOG_INFO("Tf Lite Interpreter Invoker failed");
            TfLiteInterpreterDelete(interpreter);
        }

        // Getting the output
        size_t outSize = TfLiteTensorByteSize(outputTensor);
        TfLiteType outType = TfLiteTensorType(outputTensor);

        if (outType == kTfLiteFloat32) {
            size_t n_floats = outSize / sizeof(float);
            float outBuff[5];
            if (TfLiteTensorCopyToBuffer(outputTensor, outBuff, outSize) != kTfLiteOk) {
                LOG_INFO("Failed to buffer the output");
                TfLiteInterpreterDelete(interpreter);
            }

            update_nsfw_score(outBuff, nsfwAggregation);

        }
        TfLiteInterpreterDelete(interpreter);
        return true;
    }

    void ImageProcessorNsfwLite::clean_up_model() {

        TfLiteInterpreterOptionsDelete(interpreterOptions);
        TfLiteModelDelete(model);
    }

    void ImageProcessorNsfwLite::update_nsfw_score(float *score, NsfwAggregation &nsfwAggregation) {
        {
            std::lock_guard guard{mutex_};
            nsfwAggregation.drawing = std::max(nsfwAggregation.drawing, score[0]);
            nsfwAggregation.hentai = std::max(nsfwAggregation.hentai, score[1]);
            nsfwAggregation.neutral = std::max(nsfwAggregation.neutral, score[2]);
            nsfwAggregation.porn = std::max(nsfwAggregation.porn, score[3]);
            nsfwAggregation.sexy = std::max(nsfwAggregation.sexy, score[4]);
        }
    }

    bool
    ImageProcessorNsfwLite::get_pixels_and_process_image_single_pass(JNIEnv *env, jobject bitmap,
                                                                     const char *modelPath,
                                                                     NsfwAggregation &nsfwAggregation) {
        AndroidBitmapInfo info;
        void *pixels;
        if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
            LOG_INFO("Failed to get bitmap info");
            return false;
        }

        if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888) {
            LOG_INFO("Invalid bit map format required RGBA 8888");
            return false;
        }
        if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
            LOG_ERROR("Failed to lock android pixels");
            return false;
        }

        uint8_t *resized_image = new uint8_t[224 * 224 * 4];
        float *tensor = new float[224 * 224 * 3];
        ImageProcessorOnnx::resize_image_bilinear_letter(reinterpret_cast<uint8_t *>(pixels),
                                                         info.width, info.height, info.stride,
                                                         resized_image, 224, 224);
        prepare_tensor_non_planar_neon(tensor, resized_image, 224, 224);
        init_model(modelPath);
        process_model(tensor, 224 * 224 * 3 * sizeof(float), modelPath, nsfwAggregation);
        clean_up_model();

        LOG_INFO("Drawing %f", nsfwAggregation.drawing);
        LOG_INFO("Hentai %f", nsfwAggregation.hentai);
        LOG_INFO("Neutral %f", nsfwAggregation.neutral);
        LOG_INFO("Porn %f", nsfwAggregation.porn);
        LOG_INFO("Sexy %f", nsfwAggregation.sexy);
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;

    }

    jobject ImageProcessorNsfwLite::get_nsfw_factors_for_java_bridge(JNIEnv *env,
                                                                     NsfwAggregation &aggregation) {
        jclass nsfwClassInJava = env->FindClass("com/os/nsfwProcessor/NsfwFactors");
        if (!nsfwClassInJava) return nullptr;
        jmethodID constructor = env->GetMethodID(nsfwClassInJava, "<init>", "()V");
        jobject nsfwObject = env->NewObject(nsfwClassInJava, constructor);

        jfieldID drawing = env->GetFieldID(nsfwClassInJava, "drawing", "F");
        jfieldID hentai = env->GetFieldID(nsfwClassInJava, "hentai", "F");
        jfieldID neutral = env->GetFieldID(nsfwClassInJava, "neutral", "F");
        jfieldID porn = env->GetFieldID(nsfwClassInJava, "porn", "F");
        jfieldID sexy = env->GetFieldID(nsfwClassInJava, "sexy", "F");

        env->SetFloatField(nsfwObject, drawing, aggregation.drawing);
        env->SetFloatField(nsfwObject, hentai, aggregation.hentai);
        env->SetFloatField(nsfwObject, neutral, aggregation.neutral);
        env->SetFloatField(nsfwObject, porn, aggregation.porn);
        env->SetFloatField(nsfwObject, sexy, aggregation.sexy);

        return nsfwObject;
    }
}

extern "C" {
JNIEXPORT jobject JNICALL
Java_com_os_nsfwProcessor_JniBridge_scanImageForNsfwSinglePass(JNIEnv *env, jclass jclazz,
                                                               jstring modelName, jobject bitmap) {
    const char *model = env->GetStringUTFChars(modelName, nullptr);
    ml::NsfwAggregation aggregation{};
    bool result = ml::ImageProcessorNsfwLite::get_pixels_and_process_image_single_pass(
            env, bitmap,
            model, aggregation);
    //ml::ImageProcessorOnnx::get_pixels_and_process_image(env, bitmap, model);
    env->ReleaseStringUTFChars(modelName, model);
    if (!result) return nullptr;

    return ml::ImageProcessorNsfwLite::get_nsfw_factors_for_java_bridge(env, aggregation);
}
}
extern "C"
JNIEXPORT jobject JNICALL
Java_com_os_nsfwProcessor_JniBridge_scanImageForNsfwDeepWindowPass(JNIEnv *env, jclass clazz,
                                                                   jstring model_name,
                                                                   jobject bitmap) {
    const char *model = env->GetStringUTFChars(model_name, nullptr);
    ml::NsfwAggregation aggregation{};
    bool result = ml::ImageProcessorNsfwLite::get_pixels_and_process_image_sliding_window(
            env, bitmap,
            model, aggregation);
    env->ReleaseStringUTFChars(model_name, model);
    if (!result) return nullptr;
    return ml::ImageProcessorNsfwLite::get_nsfw_factors_for_java_bridge(env, aggregation);
}