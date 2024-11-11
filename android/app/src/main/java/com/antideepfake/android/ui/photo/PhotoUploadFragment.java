package com.antideepfake.android.ui.photo;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;

import com.antideepfake.android.databinding.FragmentUploadBinding;

import java.io.IOException;

public class PhotoUploadFragment extends Fragment {

    private FragmentUploadBinding binding;
    private final ActivityResultLauncher<Intent> pickImageLauncher = registerForActivityResult(
            new ActivityResultContracts.StartActivityForResult(),
            result -> {
                if (result.getResultCode() == getActivity().RESULT_OK && result.getData() != null) {
                    Uri imageUri = result.getData().getData();
                    try {
                        Bitmap bitmap = MediaStore.Images.Media.getBitmap(getActivity().getContentResolver(), imageUri);
                        binding.imageView.setImageBitmap(bitmap);  // binding을 사용하여 ImageView에 접근
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            });

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentUploadBinding.inflate(inflater, container, false);  // View Binding 객체 생성
        return binding.getRoot();  // 바인딩 객체의 루트 뷰 반환
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        // binding을 사용하여 Button에 OnClickListener 설정
        binding.uploadButton.setOnClickListener(v -> openImagePicker());
    }

    private void openImagePicker() {
        Intent intent = new Intent();
        intent.setType("image/*");
        intent.setAction(Intent.ACTION_GET_CONTENT);
        pickImageLauncher.launch(Intent.createChooser(intent, "Select Picture"));
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;  // 메모리 누수를 방지하기 위해 뷰가 파괴될 때 binding을 null로 설정
    }
}
