package com.antideepfake.android;

import android.os.Bundle;
import android.view.View;
import android.view.animation.Animation;
import android.view.animation.AnimationUtils;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.NavigationUI;

import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // BottomNavigationView 설정
        BottomNavigationView navView = findViewById(R.id.nav_view);

        // "Upload Photo" 아이템에 애니메이션 적용
//        View uploadItem = navView.findViewById(R.id.navigation_upload);
//        Animation bounceAnimation = AnimationUtils.loadAnimation(this, R.anim.bounce);
//        uploadItem.startAnimation(bounceAnimation);

        // 네비게이션을 구성하고 NavController 설정
        // (기본적으로 NavController 설정을 위한 코드가 추가되어야 함)
        // NavHostFragment를 사용하여 네비게이션 그래프 연결
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_activity_main);
        NavigationUI.setupWithNavController(navView, navController);

    }
}
