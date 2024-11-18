package com.antideepfake.android.utils;

import android.content.Context;
import android.content.SharedPreferences;

public class SharedPreferencesHelper {

    private static final String PREFS_NAME = "antideepfake_prefs";
    private static final String CONSENT_KEY = "consent_given";

    private final SharedPreferences sharedPreferences;

    public SharedPreferencesHelper(Context context) {
        this.sharedPreferences = context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE);
    }

    public boolean isConsentGiven() {
        return sharedPreferences.getBoolean(CONSENT_KEY, false);
    }

    public void setConsentGiven(boolean consent) {
        sharedPreferences.edit().putBoolean(CONSENT_KEY, consent).commit();
    }
}
