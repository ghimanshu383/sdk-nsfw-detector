package com.os.nsfwProcessor;

import android.graphics.Bitmap;

public class JniBridge {
    static {
        System.loadLibrary("NsfwMl");
    }

    public static native NsfwFactors scanImageForNsfwSinglePass(String modelName, Bitmap bitmap);
    public static native NsfwFactors scanImageForNsfwDeepWindowPass(String modelName, Bitmap bitmap);
}
