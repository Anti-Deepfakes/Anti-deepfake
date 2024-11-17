package com.antideepfake.android.ui.dashboard;

import android.content.ContentResolver;
import android.database.Cursor;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageView;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.DialogFragment;

import com.antideepfake.android.R;
import com.bumptech.glide.Glide;

public class ImageDetailsDialogFragment extends DialogFragment {

    private static final String TAG = "ImageDetailsDialog";
    private static final String ARG_IMAGE_URI = "imageUri";

    public static ImageDetailsDialogFragment newInstance(String imageUri) {
        ImageDetailsDialogFragment fragment = new ImageDetailsDialogFragment();
        Bundle args = new Bundle();
        args.putString(ARG_IMAGE_URI, imageUri);
        fragment.setArguments(args);
        return fragment;
    }

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        return inflater.inflate(R.layout.fragment_image_details, container, false);
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        ImageView imageView = view.findViewById(R.id.imageView);
        TextView createdDateView = view.findViewById(R.id.createdDateView);

        if (getArguments() != null) {
            String imageUriString = getArguments().getString(ARG_IMAGE_URI);
            Uri imageUri = Uri.parse(imageUriString);
            Log.d(TAG, "Displaying image: " + imageUri);

            // Load the image with Glide
            Glide.with(this)
                    .load(imageUri)
                    .into(imageView);

            // Get and display creation date
            String createdDate = getCreatedDate(imageUri);
            createdDateView.setText(createdDate != null ? "생성일: " + createdDate : "날짜를 불러올 수 없습니다.");
        }

        // Close dialog when clicked outside or on image
        view.setOnClickListener(v -> dismiss());
    }

    private String getCreatedDate(Uri uri) {
        ContentResolver contentResolver = requireContext().getContentResolver();
        String[] projection = {MediaStore.Images.Media.DATE_ADDED};
        Cursor cursor = contentResolver.query(uri, projection, null, null, null);

        if (cursor != null) {
            if (cursor.moveToFirst()) {
                long dateAdded = cursor.getLong(cursor.getColumnIndexOrThrow(MediaStore.Images.Media.DATE_ADDED));
                cursor.close();
                return android.text.format.DateFormat.format("yyyy-MM-dd HH:mm:ss", dateAdded * 1000).toString();
            }
            cursor.close();
        }
        return null;
    }
}
