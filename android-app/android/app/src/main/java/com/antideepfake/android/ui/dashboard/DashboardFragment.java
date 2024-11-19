package com.antideepfake.android.ui.dashboard;

import android.Manifest;
import android.content.ContentResolver;
import android.content.ContentUris;
import android.content.pm.PackageManager;
import android.database.Cursor;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;

import com.antideepfake.android.databinding.FragmentDashboardBinding;

import java.util.ArrayList;
import java.util.List;

public class DashboardFragment extends Fragment {

    private static final String TAG = "DashboardFragment";
    private FragmentDashboardBinding binding;

    // 권한 요청 런처
    private final ActivityResultLauncher<String> permissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted) {
                    loadImages(); // 권한 허용 시 이미지 로드
                }
            });

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentDashboardBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        setupRecyclerView();
        checkAndRequestPermissions(); // 권한 확인 및 요청
    }

    private void setupRecyclerView() {
        binding.recyclerView.setLayoutManager(new GridLayoutManager(getContext(), 3));
        binding.recyclerView.setHasFixedSize(true);
    }

    private void checkAndRequestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_MEDIA_IMAGES)
                    == PackageManager.PERMISSION_GRANTED) {
                loadImages(); // 권한이 이미 허용된 경우
            } else {
                permissionLauncher.launch(Manifest.permission.READ_MEDIA_IMAGES); // 권한 요청
            }
        } else {
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE)
                    == PackageManager.PERMISSION_GRANTED) {
                loadImages(); // 권한이 이미 허용된 경우
            } else {
                permissionLauncher.launch(Manifest.permission.READ_EXTERNAL_STORAGE); // 권한 요청
            }
        }
    }

    private void loadImages() {
        List<Uri> imageUris = getImagesFromGallery("antideepfake");
        ImageAdapter adapter = new ImageAdapter(getContext(), imageUris);
        binding.recyclerView.setAdapter(adapter);

        // 이미지 클릭 이벤트 처리
        adapter.setOnItemClickListener(this::openImageDetails);

        Log.d(TAG, "이미지 로드 완료, 갤러리 이미지 개수: " + imageUris.size());
    }

    private List<Uri> getImagesFromGallery(String folderName) {
        List<Uri> imageUris = new ArrayList<>();
        ContentResolver contentResolver = requireContext().getContentResolver();
        Uri collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String selection = MediaStore.Images.Media.RELATIVE_PATH + " LIKE ? AND " +
                MediaStore.Images.Media.MIME_TYPE + " IN ('image/jpeg', 'image/png')";
        String[] selectionArgs = new String[]{"%" + folderName + "%"};
        String sortOrder = MediaStore.Images.Media.DATE_ADDED + " DESC"; // 최신순 정렬

        try (Cursor cursor = contentResolver.query(
                collection,
                new String[]{MediaStore.Images.Media._ID},
                selection,
                selectionArgs,
                sortOrder // 정렬 조건 추가
        )) {
            if (cursor != null) {
                while (cursor.moveToNext()) {
                    long id = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Images.Media._ID));
                    Uri contentUri = ContentUris.withAppendedId(collection, id);
                    imageUris.add(contentUri);
                }
            }
        } catch (Exception e) {
            Log.e(TAG, "Error reading images from gallery", e);
        }
        Log.d(TAG, "이미지 URI 목록: " + imageUris);
        return imageUris;
    }

    private void openImageDetails(Uri imageUri) {
        ImageDetailsDialogFragment dialogFragment = ImageDetailsDialogFragment.newInstance(imageUri.toString());
        dialogFragment.show(requireActivity().getSupportFragmentManager(), "ImageDetailsDialog");
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
