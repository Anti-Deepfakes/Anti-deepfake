package com.antideepfake.android.ui.photo;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Intent;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Base64;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.antideepfake.android.databinding.FragmentUploadBinding;
import com.antideepfake.android.utils.ImageUtils;

import org.json.JSONObject;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.concurrent.TimeUnit;

import okhttp3.Call;
import okhttp3.Callback;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class PhotoUploadFragment extends Fragment {

    private static final String TAG = "PhotoUploadFragment";
    private static final String SERVER_URL = "https://anti-deepfake.kr/disrupt/disrupt/generate";

    private FragmentUploadBinding binding;
    private Bitmap transformedBitmap; // 서버에서 변환된 이미지를 저장할 Bitmap
    private String originalFileName; // 선택한 이미지의 원본 파일 이름

    private final ActivityResultLauncher<Intent> pickImageLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == getActivity().RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        // 이미지 이름 가져오기
                        originalFileName = getFileNameFromUri(imageUri);

                        // 이미지 로드 및 서버 업로드
                        Bitmap bitmap = ImageUtils.loadBitmapAndCorrectOrientation(getActivity(), imageUri);
                        uploadImageToServer(bitmap);
                    } catch (IOException e) {
                        Log.e(TAG, "이미지 불러오기 실패", e);
                        Toast.makeText(getActivity(), "이미지 불러오기 실패", Toast.LENGTH_SHORT).show();
                    }
                }
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentUploadBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // 이미지 업로드 버튼 클릭 이벤트
        binding.uploadButton.setOnClickListener(v -> openImagePicker());

        // 이미지 저장 버튼 클릭 이벤트
        binding.saveButton.setOnClickListener(v -> {
            if (transformedBitmap != null) {
                saveImageToGallery(transformedBitmap, originalFileName);
            } else {
                Toast.makeText(getActivity(), "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void openImagePicker() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        pickImageLauncher.launch(Intent.createChooser(intent, "이미지를 선택하세요"));
    }

    private String getFileNameFromUri(Uri uri) {
        ContentResolver contentResolver = requireActivity().getContentResolver();
        Cursor cursor = contentResolver.query(uri, null, null, null, null);
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

    private void uploadImageToServer(Bitmap bitmap) {
        File tempFile = createTempFile(bitmap);
        if (tempFile == null) {
            Toast.makeText(getActivity(), "이미지 파일 생성 실패", Toast.LENGTH_SHORT).show();
            return;
        }

        OkHttpClient client = new OkHttpClient.Builder()
                .connectTimeout(120, TimeUnit.SECONDS) // 연결 타임아웃 120초
                .readTimeout(120, TimeUnit.SECONDS)    // 읽기 타임아웃 120초
                .writeTimeout(120, TimeUnit.SECONDS)   // 쓰기 타임아웃 120초
                .build();

        RequestBody requestBody = new MultipartBody.Builder()
                .setType(MultipartBody.FORM)
                .addFormDataPart("image", tempFile.getName(),
                        RequestBody.create(tempFile, MediaType.parse("image/jpeg")))
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
                requireActivity().runOnUiThread(() ->
                        Toast.makeText(getActivity(), "이미지 업로드 실패", Toast.LENGTH_SHORT).show());
            }

            @Override
            public void onResponse(@NonNull Call call, @NonNull Response response) throws IOException {
                if (!response.isSuccessful()) {
                    Log.e(TAG, "서버 응답 에러: " + response.code());
                    requireActivity().runOnUiThread(() ->
                            Toast.makeText(getActivity(), "서버 에러", Toast.LENGTH_SHORT).show());
                    return;
                }

                String responseBody = response.body().string();
                handleServerResponse(responseBody);
            }
        });
    }

    private File createTempFile(Bitmap bitmap) {
        try {
            File tempFile = File.createTempFile("upload_image", ".jpg", requireActivity().getCacheDir());
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos);
            }
            return tempFile;
        } catch (IOException e) {
            Log.e(TAG, "임시 파일 생성 실패", e);
            return null;
        }
    }

    private void handleServerResponse(String responseBody) {
        try {
            JSONObject jsonResponse = new JSONObject(responseBody);
            String base64Image = jsonResponse.getString("data");

            byte[] decodedString = Base64.decode(base64Image, Base64.DEFAULT);
            transformedBitmap = BitmapFactory.decodeByteArray(decodedString, 0, decodedString.length);

            requireActivity().runOnUiThread(() -> {
                binding.imageView.setImageBitmap(transformedBitmap);
                Toast.makeText(getActivity(), "이미지 변환 성공", Toast.LENGTH_SHORT).show();
            });
        } catch (Exception e) {
            Log.e(TAG, "서버 응답 처리 실패", e);
            requireActivity().runOnUiThread(() ->
                    Toast.makeText(getActivity(), "서버 응답 처리 실패", Toast.LENGTH_SHORT).show());
        }
    }

    private void saveImageToGallery(Bitmap bitmap, String originalFileName) {
        // 원본 파일 이름에 "transformed_" 접두어 추가
        String fileName = "transformed_" + originalFileName;

        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/antideepfake");

        Uri uri = getActivity().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        try {
            if (uri != null) {
                try (FileOutputStream out = (FileOutputStream) getActivity().getContentResolver().openOutputStream(uri)) {
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    out.flush();
                    Toast.makeText(getActivity(), "이미지가 갤러리에 저장되었습니다: " + fileName, Toast.LENGTH_SHORT).show();
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "이미지 저장 실패", e);
            Toast.makeText(getActivity(), "이미지 저장에 실패했습니다.", Toast.LENGTH_SHORT).show();
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
