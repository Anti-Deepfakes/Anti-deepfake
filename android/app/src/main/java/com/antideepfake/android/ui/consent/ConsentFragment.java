package com.antideepfake.android.ui.consent;

import android.os.Bundle;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.navigation.Navigation;

import com.antideepfake.android.R;
import com.antideepfake.android.databinding.FragmentConsentBinding;
import com.antideepfake.android.utils.SharedPreferencesHelper;

public class ConsentFragment extends Fragment {

    private FragmentConsentBinding binding;
    private SharedPreferencesHelper sharedPreferencesHelper;

    @Nullable
    @Override
    public View onCreateView(@NonNull LayoutInflater inflater, @Nullable ViewGroup container, @Nullable Bundle savedInstanceState) {
        binding = FragmentConsentBinding.inflate(inflater, container, false);
        sharedPreferencesHelper = new SharedPreferencesHelper(requireContext());
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        binding.submitButton.setOnClickListener(v -> {
            int selectedId = binding.consentRadioGroup.getCheckedRadioButtonId();
            if (selectedId == -1) {
                Toast.makeText(getActivity(), "동의 여부를 선택해주세요.", Toast.LENGTH_SHORT).show();
            } else if (selectedId == R.id.agreeButton) {
                // 동의 상태 저장
                sharedPreferencesHelper.setConsentGiven(true);
                Toast.makeText(getActivity(), "동의해주셔서 감사합니다.", Toast.LENGTH_SHORT).show();
                // Navigation Component를 통해 PhotoUploadFragment로 이동
                Navigation.findNavController(view).navigate(R.id.navigation_upload);
            } else if (selectedId == R.id.disagreeButton) {
                Toast.makeText(getActivity(), "동의하지 않으면 서비스를 사용할 수 없습니다.", Toast.LENGTH_SHORT).show();
            }
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}
