package com.antideepfake.android;

import android.app.Application;
import android.util.Log;

import androidx.work.ExistingPeriodicWorkPolicy;
import androidx.work.PeriodicWorkRequest;
import androidx.work.WorkManager;

import com.antideepfake.android.worker.PhotoTransformationWorker;

import java.util.concurrent.TimeUnit;

public class AntiDeepfake extends Application {

    private static final String TAG = "AntiDeepfakeApp";

    @Override
    public void onCreate() {
        super.onCreate();
        schedulePhotoTransformationWork(); // 앱 시작 시 WorkManager 작업 스케줄링
    }

    private void schedulePhotoTransformationWork() {
        Log.d(TAG, "15분 주기로 PhotoTransformationWorker 스케줄링 시작");

        PeriodicWorkRequest transformationWork =
                new PeriodicWorkRequest.Builder(PhotoTransformationWorker.class, 15, TimeUnit.MINUTES)
                        .build();

        WorkManager.getInstance(this).enqueueUniquePeriodicWork(
                "PhotoTransformationWork",
                ExistingPeriodicWorkPolicy.KEEP, // 기존 작업이 있으면 유지
                transformationWork
        );

        Log.d(TAG, "PhotoTransformationWorker 스케줄링 설정 완료");
    }
}
