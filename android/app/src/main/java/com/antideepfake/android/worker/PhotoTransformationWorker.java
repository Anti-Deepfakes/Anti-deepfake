package com.antideepfake.android.worker;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.antideepfake.android.utils.ImageUtils;

import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

public class PhotoTransformationWorker extends Worker {

    private static final String TAG = "PhotoTransformationWorker";
    private static final String TARGET_FOLDER = Environment.DIRECTORY_PICTURES + "/antideepfake";

    public PhotoTransformationWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        try {
            // 최근 갤러리에 추가된 이미지 가져오기
            List<Uri> recentImages = getRecentImages();
            for (Uri imageUri : recentImages) {
                String originalFileName = getFileNameFromUri(imageUri);

                if (originalFileName == null) {
                    Log.e(TAG, "파일 이름을 가져오지 못했습니다.");
                    continue;
                }

                // antideepfake 폴더에 동일한 이름의 파일이 있는지 확인하여 이미 변환된 경우 건너뛰기
                if (isImageAlreadyTransformed(originalFileName)) {
                    Log.d(TAG, "이미 변환된 이미지입니다: " + originalFileName);
                    continue;
                }

                // 이미지 로드
//                Bitmap bitmap = MediaStore.Images.Media.getBitmap(getApplicationContext().getContentResolver(), imageUri);
                Bitmap bitmap = ImageUtils.loadBitmapAndCorrectOrientation(getApplicationContext(), imageUri);
                // 흑백 변환 적용
                Bitmap transformedBitmap = applyTransformation(bitmap);

                // 변환된 이미지를 antideepfake 폴더에 저장
                saveImageToGallery(transformedBitmap, originalFileName);
                Log.d(TAG, "이미지를 변환하여 갤러리에 저장 완료: " + originalFileName);
            }
            return Result.success();
        } catch (Exception e) {
            Log.e(TAG, "이미지 변환 중 오류 발생", e);
            return Result.failure();
        }
    }

    // MediaStore에서 최근에 추가된 이미지들 중 antideepfake 폴더를 제외한 목록을 가져오는 메서드
    private List<Uri> getRecentImages() {
        List<Uri> imageUris = new ArrayList<>();
        Uri collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String selection = MediaStore.Images.Media.RELATIVE_PATH + " NOT LIKE ?";
        String[] selectionArgs = new String[]{"%" + TARGET_FOLDER + "%"}; // antideepfake 폴더 제외

        Cursor cursor = getApplicationContext().getContentResolver().query(
                collection,
                new String[]{MediaStore.Images.Media._ID},
                selection,
                selectionArgs,
                null
        );

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

    // URI에서 파일 이름을 가져오는 메서드
    private String getFileNameFromUri(Uri uri) {
        Cursor cursor = getApplicationContext().getContentResolver().query(uri, null, null, null, null);
        if (cursor != null) {
            int nameIndex = cursor.getColumnIndex(MediaStore.Images.Media.DISPLAY_NAME);
            if (cursor.moveToFirst()) {
                String fileName = cursor.getString(nameIndex);
                cursor.close();
                return fileName;
            }
            cursor.close();
        }
        return null;
    }

    // antideepfake 폴더에 동일한 파일 이름이 존재하는지 확인
    private boolean isImageAlreadyTransformed(String fileName) {
        File targetDir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_PICTURES), "antideepfake");
        File[] files = targetDir.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.getName().equals(fileName)) {
                    return true;
                }
            }
        }
        return false;
    }

    // 흑백 변환 메서드
    //TODO 모델 호출 코드로 변경
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

    // 변환된 이미지를 antideepfake 폴더에 저장
    private void saveImageToGallery(Bitmap bitmap, String originalFileName) {
        Log.d(TAG, "saveImageToGallery() 호출됨");

        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, originalFileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, TARGET_FOLDER);

        Uri uri = getApplicationContext().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        try {
            if (uri != null) {
                try (OutputStream out = getApplicationContext().getContentResolver().openOutputStream(uri)) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    Log.d(TAG, "이미지가 antideepfake 폴더에 저장되었습니다.");
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "이미지 저장 중 오류 발생", e);
        }
    }
}
