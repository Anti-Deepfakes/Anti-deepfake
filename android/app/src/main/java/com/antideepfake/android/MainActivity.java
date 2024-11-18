package com.antideepfake.android;

import android.os.Bundle;
import android.view.View;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.navigation.NavController;
import androidx.navigation.Navigation;
import androidx.navigation.ui.NavigationUI;

import com.antideepfake.android.utils.SharedPreferencesHelper;
import com.google.android.material.bottomnavigation.BottomNavigationView;

public class MainActivity extends AppCompatActivity {

    private SharedPreferencesHelper sharedPreferencesHelper;

    @Override
    protected void onCreate(@Nullable Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        // BottomNavigationView 설정
        BottomNavigationView navView = findViewById(R.id.nav_view);
        NavController navController = Navigation.findNavController(this, R.id.nav_host_fragment_activity_main);
        sharedPreferencesHelper = new SharedPreferencesHelper(this);

        NavigationUI.setupWithNavController(navView, navController);

        boolean isConsentGiven = sharedPreferencesHelper.isConsentGiven();

        // 동의 여부에 따라 초기 화면 설정
        if (!isConsentGiven && navController.getCurrentDestination() != null
                && navController.getCurrentDestination().getId() != R.id.consentFragment) {
            navController.navigate(R.id.consentFragment);
            navView.setVisibility(View.GONE); // 동의 전 BottomNavigation 숨기기
        }

        // Fragment 변경 시 BottomNavigationView 상태 업데이트
        navController.addOnDestinationChangedListener((controller, destination, arguments) -> {
            if (destination.getId() == R.id.consentFragment) {
                navView.setVisibility(View.GONE); // ConsentFragment에서는 숨기기
            } else {
                navView.setVisibility(View.VISIBLE); // 다른 Fragment에서는 보이기
            }
        });

        // BottomNavigationView 클릭 리스너 설정
        navView.setOnItemSelectedListener(item -> {
            // 동의하지 않았다면 네비게이션 제한
            if (!sharedPreferencesHelper.isConsentGiven()) {
                Toast.makeText(this, "동의 후에 이동 가능합니다.", Toast.LENGTH_SHORT).show();
                return false;
            }
            // 네비게이션 처리
            return NavigationUI.onNavDestinationSelected(item, navController);
        });

    }
}
