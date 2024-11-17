package com.antideepfake.android.worker;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.annotation.NonNull;
import androidx.work.Worker;
import androidx.work.WorkerParameters;

import com.antideepfake.android.utils.ImageUtils;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class PhotoTransformationWorker extends Worker {

    private static final String TAG = "PhotoTransformationWorker";
    private static final String TARGET_FOLDER = Environment.DIRECTORY_PICTURES + "/antideepfake";
    private static final String SERVER_URL = "https://anti-deepfake.kr/disrupt/disrupt/generate";

    public PhotoTransformationWorker(@NonNull Context context, @NonNull WorkerParameters workerParams) {
        super(context, workerParams);
    }

    @NonNull
    @Override
    public Result doWork() {
        try {
            List<Uri> recentImages = getRecentImages();
            for (Uri imageUri : recentImages) {
                String originalFileName = getFileNameFromUri(imageUri);

                if (originalFileName == null) {
                    Log.e(TAG, "파일 이름을 가져오지 못했습니다.");
                    continue;
                }

                if (isImageAlreadyTransformed(originalFileName)) {
                    Log.d(TAG, "이미 변환된 이미지입니다: " + originalFileName);
                    continue;
                }

                Bitmap bitmap = ImageUtils.loadBitmapAndCorrectOrientation(getApplicationContext(), imageUri);

                detectFaces(bitmap, originalFileName);
            }
            return Result.success();
        } catch (Exception e) {
            Log.e(TAG, "이미지 처리 중 오류 발생", e);
            return Result.failure();
        }
    }

    private List<Uri> getRecentImages() {
        List<Uri> imageUris = new ArrayList<>();
        Uri collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String selection = MediaStore.Images.Media.RELATIVE_PATH + " NOT LIKE ?";
        String[] selectionArgs = new String[]{"%" + TARGET_FOLDER + "%"};

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

    private void detectFaces(Bitmap bitmap, String originalFileName) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .build();

        FaceDetector detector = FaceDetection.getClient(options);

        detector.process(image)
                .addOnSuccessListener(faces -> {
                    if (faces.size() > 0) {
                        Log.d(TAG, "얼굴 감지됨, 서버에 요청 시작");
                        uploadImageToServer(bitmap, originalFileName);
                    } else {
                        Log.d(TAG, "얼굴이 감지되지 않아 변환을 건너뜀");
                    }
                })
                .addOnFailureListener(e -> Log.e(TAG, "얼굴 감지 실패", e));
    }

    private void uploadImageToServer(Bitmap bitmap, String originalFileName) {
        File tempFile = createTempFile(bitmap);
        if (tempFile == null) {
            Log.e(TAG, "임시 파일 생성 실패");
            return;
        }

        OkHttpClient client = new OkHttpClient();
        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", tempFile.getName(), RequestBody.create(tempFile, MediaType.parse("image/jpeg")))
                .build();

        Request request = new Request.Builder()
                .url(SERVER_URL)
                .post(requestBody)
                .addHeader("Accept", "application/json")
                .build();

        client.newCall(request).enqueue(new Callback() {
            @Override
            public void onFailure(@NonNull Call call, @NonNull IOException e) {
                Log.e(TAG, "이미지 업로드 실패", e);
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(TAG, "서버 응답 에러: " + response.code());
                    return;
                }

                String responseBody = response.body().string();
                handleServerResponse(responseBody, originalFileName);
            }
        });
    }

    private File createTempFile(Bitmap bitmap) {
        try {
            File tempFile = File.createTempFile("upload_image", ".jpg", getApplicationContext().getCacheDir());
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            }
            return tempFile;
        } catch (IOException e) {
            Log.e(TAG, "임시 파일 생성 실패", e);
            return null;
        }
    }

    private void handleServerResponse(String responseBody, String originalFileName) {
        try {
            JSONObject jsonResponse = new JSONObject(responseBody);
            String base64Image = jsonResponse.getString("data");
            byte[] decodedBytes = android.util.Base64.decode(base64Image, android.util.Base64.DEFAULT);
            Bitmap transformedBitmap = android.graphics.BitmapFactory.decodeByteArray(decodedBytes, 0, decodedBytes.length);

            saveImageToGallery(transformedBitmap, originalFileName);
            Log.d(TAG, "서버 응답 처리 완료: " + originalFileName);
        } catch (Exception e) {
            Log.e(TAG, "서버 응답 처리 실패", e);
        }
    }

    private void saveImageToGallery(Bitmap bitmap, String originalFileName) {
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, originalFileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, TARGET_FOLDER);

        Uri uri = getApplicationContext().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        try {
            if (uri != null) {
                try (OutputStream out = getApplicationContext().getContentResolver().openOutputStream(uri)) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    Log.d(TAG, "이미지가 갤러리에 저장되었습니다.");
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "이미지 저장 중 오류 발생", e);
        }
    }
}
