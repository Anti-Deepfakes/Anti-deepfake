package com.antideepfake.android.ui.dashboard;

import android.os.Bundle;
import android.os.Environment;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.fragment.app.Fragment;
import androidx.recyclerview.widget.GridLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import com.antideepfake.android.databinding.FragmentDashboardBinding;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class DashboardFragment extends Fragment {

    private FragmentDashboardBinding binding;

    @Override
    public View onCreateView(@NonNull LayoutInflater inflater,
                             ViewGroup container, Bundle savedInstanceState) {

        binding = FragmentDashboardBinding.inflate(inflater, container, false);
        View root = binding.getRoot();

        RecyclerView recyclerView = binding.recyclerView;
        recyclerView.setLayoutManager(new GridLayoutManager(getContext(), 3)); // 3열의 GridLayout

        List<File> imageFiles = getImagesFromGallery("antideepfake");
        ImageAdapter adapter = new ImageAdapter(getContext(), imageFiles);
        recyclerView.setAdapter(adapter);

        return root;
    }

    private List<File> getImagesFromGallery(String folderName) {
        List<File> imageFiles = new ArrayList<>();
        File galleryFolder = new File(Environment.getExternalStorageDirectory(), folderName);

        if (galleryFolder.exists() && galleryFolder.isDirectory()) {
            File[] files = galleryFolder.listFiles();
            if (files != null) {
                for (File file : files) {
                    if (file.isFile() && isImageFile(file)) {
                        imageFiles.add(file);
                    }
                }
            }
        }
        return imageFiles;
    }

    private boolean isImageFile(File file) {
        String[] extensions = {"jpg", "jpeg", "png", "bmp", "gif"};
        for (String ext : extensions) {
            if (file.getName().toLowerCase().endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
