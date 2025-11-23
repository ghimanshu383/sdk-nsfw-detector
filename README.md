# Sdk-nsfw-detector

This repository contains the native and Kotlin code for an offline NSFW image detection SDK for Android.
The goal of the SDK is to provide a small, fast, dependency-free module that can classify images locally without requiring any network connectivity.

The implementation uses:

- Android NDK (C++17)

- ARM NEON SIMD optimizations for tensor preparation

- TFLite for running the model

- Simple JNI bridge for calling the native functions from Kotlin

- Optional sliding-window “deep scan” for partial-image detection

- The SDK exposes two Kotlin APIs:

  - Single-Pass Scan – fast full-image NSFW classification

  - Deep Window Scan – slower, performs sliding-window analysis across the image

**This repo includes both the source code (native + Kotlin) and a compiled AAR library for direct integration.**

## 1. Overview

The SDK performs classification using a small TensorFlow Lite model (included inside the SDK).
Images are passed as Bitmap objects from Kotlin.
The native layer receives the pixel buffer through AndroidBitmap_lockPixels, prepares the input tensor using an optimized NEON implementation, and runs inference.

The single-pass approach is intended for normal usage.
The window-based deep scan is optional and mainly useful for detecting localized NSFW regions in larger images (**though the model is not optmized for such scans integrating the yolo model for the same in later version**)

## 2. Features
### Single-Pass Detector

- Resizes the input image to 224×224

- Converts RGBA → normalized RGB tensor

- Uses NEON SIMD instructions (vld4q, vmovl, vcvtq) for fast preprocessing

- Runs the model once (fastest path)

**Very low latency, suitable for UI threads if called through coroutines**

### Deep Scan (Sliding Window)

- Divides the image into overlapping 224×224 windows

- Uses a stride of 112px

- Each window is processed in parallel through a small thread pool

- Suitable for images with multiple regions or partial NSFW content

**Significantly heavier than the single pass (not recommended for frequent usage)**

## 3. Native Pipeline

The native processing path looks like this:

1. **Lock Bitmap Pixels (RGBA8888)**

2. **Optional resize (bilinear + letterbox)**

3. **Extract window (for deep scan)**

4. **Tensor preparation using NEON**

5. **RGBA → R,G,B normalized float32**

6. **Non-planar memory layout**

7. **16-pixel SIMD batches**

8. **Invoke TFLite interpreter**

9. **Copy output scores**

10. **Aggregation (max score per class)**

11. **Unlock Bitmap**

The NEON path replaces manual loops with SIMD intrinsics and batches 16 pixels at a time.

## 4. Available Classes
NsfwFactors

Simple data class holding the 5 output scores:

- drawing

- hentai

- neutral

- porn

- sexy

### NsfwDetector (Kotlin API)

Contains:

suspend fun detectNsfwSinglePass(context: Context, bitmap: Bitmap): NsfwFactors
suspend fun detectNsfwDeepScanningWindows(context: Context, bitmap: Bitmap): NsfwFactors


**The model is copied from assets to the app’s internal storage on first use.**

## 6. NEON Tensor Conversion

The NEON implementation handles:

- Vectorized load of 16 RGBA pixels (vld4q_u8)

- Expanding u8 → u16 → u32

- Converting to float and multiplying by 1/255

- Interleaving R,G,B into non-planar layout for TFLite

- Each batch generates 16×3 float32 values.

- Fallback scalar code handles remaining pixels at the end of each row.


## 7. Sliding Window Details

For deep scanning:

- Window size: 224×224

- Stride: 112

- Thread pool size = hardware_concurrency

- Each thread:

  - Allocates its own window buffer

  - Allocates its own tensor buffer

  - Processes windows in its assigned range

**Scores are aggregated using a shared mutex.**

## 8. Using the SDK (AAR)

Add the AAR to your app:
- Create a lib folder or any other folder and put the aar file inside that folder
- implementation(files("<you_folder>/native-nsfw-processor-sdk-1.0.0.aar"))

Then call:

CoroutineScope(Dispatchers.IO).launch {
    val result = NsfwDetector.detectNsfwSinglePass(context, bitmap)
    Log.d("NSFW", "Porn score = ${result.porn}")
}

 The result will be a class NsfwFactors which contains different parameters results for drawing, hentai, neutral, porn, sexy ***(the drawing and neutral are nsfw safe scores and should be added to compute the total safe score)***

## 9. Model

The included model is based on the open-source work from the nsfw_model repository by GantMan:
https://github.com/GantMan/nsfw_model

Model is redistributed according to its Apache license.
Model files included inside model_used/.

## 10. Notes & Limitations

Sliding-window detection is very slow; should always be called from a background coroutine.

The model is not perfect and may produce false positives on certain types of images.

**ARMv7 NEON support is removed due to compiler issues (only arm64-v8a supported in this version).**

**No external ML dependencies except TFLite runtime.**

## 11. License
Apache License 2.0


Refer to the LICENSE file.

## 12. Future Work

- Reduce false positives on human faces in deep scan mode

- ARMv7 NEON fallback

- Optionally switch to ONNX-based models if required

- Benchmark page and example app improvements




