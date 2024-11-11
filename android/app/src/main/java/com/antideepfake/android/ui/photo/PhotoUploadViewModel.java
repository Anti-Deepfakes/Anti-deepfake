package com.antideepfake.android.ui.photo;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class PhotoUploadViewModel extends ViewModel {
    private final MutableLiveData<String> mText;

    public PhotoUploadViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("This is photo Upload fragment");
    }

    public LiveData<String> getText() {
        return mText;
    }
}
