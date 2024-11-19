package com.antideepfake.android.utils;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.media.ExifInterface;
import android.net.Uri;
import android.util.Log;

import java.io.IOException;
import java.io.InputStream;

public class ImageUtils {
    private static final String TAG = "ImageUtils";

    // EXIF 정보를 기반으로 이미지를 회전시키는 메서드
    public static Bitmap loadBitmapAndCorrectOrientation(Context context, Uri imageUri) throws IOException {
        InputStream inputStream = context.getContentResolver().openInputStream(imageUri);
        Bitmap bitmap = BitmapFactory.decodeStream(inputStream);
        inputStream.close();

        InputStream exifInputStream = context.getContentResolver().openInputStream(imageUri);
        ExifInterface exif = new ExifInterface(exifInputStream);
        int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
        exifInputStream.close();

        Log.d(TAG, "이미지 회전 정보 (orientation): " + orientation);

        switch (orientation) {
            case ExifInterface.ORIENTATION_ROTATE_90:
                return rotateBitmap(bitmap, 90);
            case ExifInterface.ORIENTATION_ROTATE_180:
                return rotateBitmap(bitmap, 180);
            case ExifInterface.ORIENTATION_ROTATE_270:
                return rotateBitmap(bitmap, 270);
            default:
                return bitmap;
        }
    }

    // Bitmap 회전 보정 메서드
    private static Bitmap rotateBitmap(Bitmap bitmap, int degrees) {
        Matrix matrix = new Matrix();
        matrix.postRotate(degrees);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }
}
