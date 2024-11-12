package com.antideepfake.android.worker;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.media.ExifInterface;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class PhotoTransformationWorker extends Worker {

    private static final String TAG = "PhotoTransformationWorker";

    public PhotoTransformationWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        try {
            // 최근 추가된 이미지 목록 가져오기
            List<Uri> recentImages = getRecentImages();
            for (Uri imageUri : recentImages) {
                Log.d(TAG, "이미지 처리 중: " + imageUri);

                // 이미지 로드
                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), imageUri);

                // Exif 정보를 통해 이미지의 회전 각도 얻기
                int rotation = getImageRotation(imageUri);
                Bitmap rotatedBitmap = rotateBitmap(bitmap, rotation);

                // 흑백 변환 적용
                Bitmap transformedBitmap = applyTransformation(rotatedBitmap);

                // 변환된 이미지를 갤러리에 저장
                saveImageToGallery(transformedBitmap);
                Log.d(TAG, "이미지를 변환하여 갤러리에 저장 완료: " + imageUri.toString());
            }
            return Result.success();
        } catch (Exception e) {
            Log.e(TAG, "이미지 변환 중 오류 발생", e);
            return Result.failure();
        }
    }

    // 최근 15분 이내에 갤러리에 추가된 이미지 목록을 가져오는 메서드
    private List<Uri> getRecentImages() {
        List<Uri> imageUris = new ArrayList<>();
        long fifteenMinutesAgo = System.currentTimeMillis() - (15 * 60 * 1000);

        // MediaStore에서 15분 이내에 추가된 이미지들만 쿼리
        Uri collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;
        String[] projection = new String[]{MediaStore.Images.Media._ID};
        String selection = MediaStore.Images.Media.DATE_ADDED + ">= ?";
        String[] selectionArgs = new String[]{String.valueOf(fifteenMinutesAgo / 1000)}; // DATE_ADDED는 초 단위

        Cursor cursor = getApplicationContext().getContentResolver().query(
                collection,
                projection,
                selection,
                selectionArgs,
                null
        );

        // 쿼리 결과에서 이미지 ID를 통해 URI 리스트 생성
        if (cursor != null) {
            int idColumn = cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID);
            while (cursor.moveToNext()) {
                long id = cursor.getLong(idColumn);
                Uri imageUri = Uri.withAppendedPath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, String.valueOf(id));
                imageUris.add(imageUri);
            }
            cursor.close();
        }

        Log.d(TAG, "처리할 최근 이미지 수: " + imageUris.size());
        return imageUris;
    }

    // Exif 정보를 통해 이미지의 회전 각도 가져오는 메서드
    private int getImageRotation(Uri imageUri) {
        try {
            ExifInterface exif = new ExifInterface(getApplicationContext().getContentResolver().openInputStream(imageUri));
            int orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL);
            switch (orientation) {
                case ExifInterface.ORIENTATION_ROTATE_90:
                    return 90;
                case ExifInterface.ORIENTATION_ROTATE_180:
                    return 180;
                case ExifInterface.ORIENTATION_ROTATE_270:
                    return 270;
                default:
                    return 0;
            }
        } catch (IOException e) {
            Log.e(TAG, "Exif 데이터 읽기 중 오류 발생", e);
            return 0;
        }
    }

    // 비트맵을 지정된 각도로 회전시키는 메서드
    private Bitmap rotateBitmap(Bitmap bitmap, int rotationAngle) {
        if (rotationAngle == 0) {
            return bitmap; // 회전이 필요 없는 경우 원본 반환
        }
        Matrix matrix = new Matrix();
        matrix.postRotate(rotationAngle);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    // 흑백 변환을 적용하는 메서드
    private Bitmap applyTransformation(Bitmap original) {
        Log.d(TAG, "흑백 변환 적용 중");
        Bitmap grayscaleBitmap = Bitmap.createBitmap(original.getWidth(), original.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0); // 흑백으로 변환
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(filter);
        canvas.drawBitmap(original, 0, 0, paint);
        return grayscaleBitmap;
    }

    // 변환된 이미지를 antideepfake 폴더에 저장하는 메서드
    private void saveImageToGallery(Bitmap bitmap) {
        Log.d(TAG, "변환된 이미지를 갤러리에 저장 중");
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, "antideepfake_" + System.currentTimeMillis() + ".jpg");
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/antideepfake");

        ContentResolver resolver = getApplicationContext().getContentResolver();
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        // 갤러리에 변환된 이미지 저장
        if (uri != null) {
            try (OutputStream out = resolver.openOutputStream(uri)) {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                Log.d(TAG, "antideepfake 폴더에 이미지 저장 완료");
            } catch (IOException e) {
                Log.e(TAG, "이미지 저장 중 오류 발생", e);
            }
        } else {
            Log.e(TAG, "이미지 저장을 위한 미디어 스토어 항목 생성 실패");
        }
    }
}
