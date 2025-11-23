package com.os.nsfwProcessor

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File

class NsfwDetector {
    companion object {
        suspend fun detectNsfwSinglePass(context: Context, bitmap: Bitmap): NsfwFactors =
            withContext(Dispatchers.Default) {
                val mutable: Bitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
                return@withContext JniBridge.scanImageForNsfwSinglePass(
                    loadModel(context, "nsfw_classifier.tflite"),
                    mutable
                )
            }

        suspend fun detectNsfwDeepScanningWindows(
            context: Context,
            bitmap: Bitmap
        ): NsfwFactors = withContext(Dispatchers.Default) {
            val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
            return@withContext JniBridge.scanImageForNsfwDeepWindowPass(
                loadModel(context, "nsfw_classifier.tflite"),
                mutable
            )
        }

        private fun loadModel(context: Context, modelPath: String): String {
            val modelFile = File(context.filesDir, modelPath);
            if (!modelFile.exists()) {
                context.assets.open(modelPath).also { input ->
                    modelFile.outputStream().also { output ->
                        input.copyTo(output)
                    }
                }
            }
            return modelFile.absolutePath;
        }
    }
}