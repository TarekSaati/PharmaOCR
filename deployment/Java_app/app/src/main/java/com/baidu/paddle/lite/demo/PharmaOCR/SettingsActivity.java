package com.baidu.paddle.lite.demo.PharmaOCR;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.EditTextPreference;

public class SettingsActivity extends AppCompatPreferenceActivity {
    private EditTextPreference etDetModelPath;
    private EditTextPreference etClsModelPath;
    private EditTextPreference etClsLabelPath;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings_ocr);

        etDetModelPath = (EditTextPreference) findPreference("DET_MODEL_PATH_KEY");
        etClsModelPath = (EditTextPreference) findPreference("CLS_MODEL_PATH_KEY");
        etClsLabelPath = (EditTextPreference) findPreference("CLS_LABEL_PATH_KEY");
    }

    private void reloadOCRPreferences() {
        SharedPreferences sp = getPreferenceScreen().getSharedPreferences();

        String detModelPath = sp.getString("DET_MODEL_PATH_KEY",
                "models/ocr_detection");
        String clsModelPath = sp.getString("CLS_MODEL_PATH_KEY",
                "models/ocr_classification");
        String clsLabelPath = sp.getString("CLS_LABEL_PATH_KEY",
                "labels/arabic_labels");

        etDetModelPath.setSummary(detModelPath);
        etClsModelPath.setSummary(clsModelPath);
        etClsLabelPath.setSummary(clsLabelPath);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sp, String key) {
        if (key.startsWith("OCR_")) {
            reloadOCRPreferences();
        }
    }
}