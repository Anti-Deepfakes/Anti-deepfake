package com.antideepfake.android.ui.home;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

public class HomeViewModel extends ViewModel {

    private final MutableLiveData<String> mText;

    public HomeViewModel() {
        mText = new MutableLiveData<>();
        mText.setValue("'가짜가 아닌 진짜를 지키다'");
    }

    public LiveData<String> getText() {
        return mText;
    }
}