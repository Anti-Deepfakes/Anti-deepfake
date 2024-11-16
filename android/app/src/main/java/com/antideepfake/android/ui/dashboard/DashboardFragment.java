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

import androidx.annotation.NonNull;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.antideepfake.android.databinding.FragmentDashboardBinding;

import java.util.ArrayList;
import java.util.List;

public class DashboardFragment extends Fragment {

    private static final String TAG = "DashboardFragment"; // Log 태그 설정
    private FragmentDashboardBinding binding;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentDashboardBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        // 권한 요청
        requestPermissions();

        // RecyclerView 설정
        RecyclerView recyclerView = binding.recyclerView;
        recyclerView.setLayoutManager(new GridLayoutManager(getContext(), 3)); // 3열의 GridLayout
        recyclerView.setHasFixedSize(true); // 크기 고정으로 성능 최적화

        // 이미지 파일 목록 가져오기
        List<Uri> imageUris = getImagesFromGallery("antideepfake");
        ImageAdapter adapter = new ImageAdapter(getContext(), imageUris);
        recyclerView.setAdapter(adapter);

        return root;
    }

    private List<Uri> getImagesFromGallery(String folderName) {
        List<Uri> imageUris = new ArrayList<>();
        ContentResolver contentResolver = requireContext().getContentResolver();
        Uri collection = MediaStore.Images.Media.EXTERNAL_CONTENT_URI;

        String selection = MediaStore.Images.Media.RELATIVE_PATH + " LIKE ? AND " +
                MediaStore.Images.Media.MIME_TYPE + " IN ('image/jpeg', 'image/png')";
        String[] selectionArgs = new String[]{"%" + folderName + "%"};

        try (Cursor cursor = contentResolver.query(
                collection,
                new String[]{MediaStore.Images.Media._ID},
                selection,
                selectionArgs,
                null
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

    private void requestPermissions() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) { // Android 13 이상
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_MEDIA_IMAGES) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(requireActivity(), new String[]{Manifest.permission.READ_MEDIA_IMAGES}, 100);
            }
        } else { // Android 12 이하
            if (ContextCompat.checkSelfPermission(requireContext(), Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(requireActivity(), new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 100);
            }
        }
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
