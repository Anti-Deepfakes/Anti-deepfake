package com.antideepfake.android.ui.photo;

import android.Manifest;
import android.content.ContentValues;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;

import com.antideepfake.android.databinding.FragmentUploadBinding;
import com.antideepfake.android.utils.ImageUtils;
import com.google.mlkit.vision.common.InputImage;
import com.google.mlkit.vision.face.FaceDetection;
import com.google.mlkit.vision.face.FaceDetector;
import com.google.mlkit.vision.face.FaceDetectorOptions;

import java.io.FileOutputStream;
import java.io.IOException;

public class PhotoUploadFragment extends Fragment {

    private static final String TAG = "PhotoUploadFragment"; // Log 태그 설정
    private FragmentUploadBinding binding;
    private Bitmap grayscaleBitmap;

    private final ActivityResultLauncher<Intent> pickImageLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == getActivity().RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        Bitmap bitmap = ImageUtils.loadBitmapAndCorrectOrientation(getActivity(), imageUri);
                        Log.d(TAG, "이미지 불러오기 및 회전 보정 성공");

                        // 흑백으로 변환하여 미리보기 ImageView에 설정
                        grayscaleBitmap = convertToGrayScale(bitmap);
                        detectFaces(bitmap);
                        binding.imageView.setImageBitmap(grayscaleBitmap);
                        Log.d(TAG, "이미지 흑백 변환 및 미리보기 설정 완료");
                    } catch (IOException e) {
                        Log.e(TAG, "이미지 불러오기 실패", e); // 예외 발생 시 로그
                        e.printStackTrace();
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

        // 권한 요청
        requestPermissions();

        // 갤러리에서 이미지 선택
        binding.uploadButton.setOnClickListener(v -> openImagePicker());

        // 흑백 변환 이미지를 저장하는 버튼 설정
        binding.disruptButton.setOnClickListener(v -> {
            if (grayscaleBitmap != null) {
                saveImageToGallery(grayscaleBitmap);
                Log.d(TAG, "이미지 저장 버튼이 눌렸고, 이미지 저장이 진행됩니다.");
            } else {
                Log.d(TAG, "저장 버튼 클릭 시 grayscaleBitmap이 null입니다.");
                Toast.makeText(getActivity(), "저장할 이미지가 없습니다.", Toast.LENGTH_SHORT).show();
            }
        });
    }

    private void openImagePicker() {
        Log.d(TAG, "openImagePicker() 호출됨");
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        pickImageLauncher.launch(Intent.createChooser(intent, "Select Picture"));
    }

    // 흑백 변환 메서드 -> TODO 모델 적용
    private Bitmap convertToGrayScale(Bitmap originalBitmap) {
        Log.d(TAG, "convertToGrayScale() 호출됨");
        Bitmap grayscaleBitmap = Bitmap.createBitmap(originalBitmap.getWidth(), originalBitmap.getHeight(), Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(grayscaleBitmap);
        Paint paint = new Paint();
        ColorMatrix colorMatrix = new ColorMatrix();
        colorMatrix.setSaturation(0); // 흑백으로 변환
        ColorMatrixColorFilter filter = new ColorMatrixColorFilter(colorMatrix);
        paint.setColorFilter(filter);
        canvas.drawBitmap(originalBitmap, 0, 0, paint);
        return grayscaleBitmap;
    }

    private void detectFaces(Bitmap bitmap) {
        InputImage image = InputImage.fromBitmap(bitmap, 0);

        FaceDetectorOptions options = new FaceDetectorOptions.Builder()
                .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_ACCURATE)
                .build();

        FaceDetector detector = FaceDetection.getClient(options);

        detector.process(image)
                .addOnSuccessListener(faces -> {
                    Log.d("FaceDetection", faces.toString());
                    int result = faces.size() > 0 ? 1 : 0; // 얼굴이 있으면 1, 없으면 0
                    onFaceDetectionResult(result);
                })
                .addOnFailureListener(e -> e.printStackTrace());
    }

    private void onFaceDetectionResult(int result) {
        if (result == 1) {
            Log.d("FaceDetection", "사람이 있습니다.");
        } else {
            Log.d("FaceDetection", "사람이 없습니다.");
        }
    }

    // 흑백 이미지를 갤러리의 antideepfake 폴더에 저장
    private void saveImageToGallery(Bitmap bitmap) {
        Log.d(TAG, "saveImageToGallery() 호출됨");
        String fileName = "gray_image_" + System.currentTimeMillis() + ".jpg";

        // 안드로이드 10 이상에서는 MediaStore 사용하여 이미지 저장
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/antideepfake");

        Uri uri = getActivity().getContentResolver().insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        try {
            if (uri != null) {
                try (FileOutputStream out = (FileOutputStream) getActivity().getContentResolver().openOutputStream(uri)) {
                    Log.d(TAG, "이미지 파일 저장 중...");
                    bitmap.compress(Bitmap.CompressFormat.JPEG, 100, out);
                    out.flush();
                    Log.d(TAG, "이미지가 갤러리에 저장되었습니다.");
                    Toast.makeText(getActivity(), "이미지가 갤러리에 저장되었습니다.", Toast.LENGTH_SHORT).show();
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "이미지 저장 중 오류 발생", e); // 오류 발생 시 로그
            Toast.makeText(getActivity(), "이미지 저장에 실패했습니다.", Toast.LENGTH_SHORT).show();
        }
    }

    // 권한 요청 메서드
    private void requestPermissions() {
        Log.d(TAG, "requestPermissions() 호출됨");
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) { // Android 14 이상
            if (ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 100);
            }
        } else { // Android 13 이하
            if (ContextCompat.checkSelfPermission(getActivity(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(getActivity(), new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
        Log.d(TAG, "onDestroyView() 호출됨, binding 해제됨");
    }
}
