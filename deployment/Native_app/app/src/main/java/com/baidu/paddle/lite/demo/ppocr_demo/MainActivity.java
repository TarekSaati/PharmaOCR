package com.baidu.paddle.lite.demo.ppocr_demo;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.preference.PreferenceManager;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.design.widget.AppBarLayout;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.ImageButton;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import com.baidu.paddle.lite.demo.common.CameraSurfaceView;
import com.baidu.paddle.lite.demo.common.Utils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Date;
import java.util.Objects;


public class MainActivity extends Activity implements View.OnClickListener, CameraSurfaceView.OnTextureChangedListener {
    private static final String TAG = MainActivity.class.getSimpleName();

    CameraSurfaceView svPreview;
    TextView tvStatus;
    ImageButton btnSwitch;
    ImageButton btnShutter;

    String savedImagePath = "result.jpg";
    int lastFrameIndex = 0;
    long lastFrameTime;

    // Model settings of object detection
    //protected String detModelPath = "detection_lite.nb";
    protected String classModelPath = "MobileNetV3Clas.nb";
    //protected String detModelPath = "ch_ppocr_mobile_v2.0_det_slim_opt.nb";
    protected String detModelPath = "Multilingual_PP-OCRv3_det_slim_infer.nb";
    protected String labelPath = "four_class_label_list";
    protected String configPath = "config.txt";
    protected int cpuThreadNum = 1;
    protected String cpuPowerMode = "LITE_POWER_HIGH";


    Native predictor = new Native();

    ArrayList<String> ocr_res;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Fullscreen
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN, WindowManager.LayoutParams.FLAG_FULLSCREEN);

        setContentView(R.layout.activity_main);


        // Init the camera preview and UI components
        initView();

        // Check and request CAMERA and WRITE_EXTERNAL_STORAGE permissions
        if (!checkAllPermissions()) {
            requestAllPermissions();
        }
        checkRun();
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.btn_switch:
                svPreview.switchCamera();
                break;
            case R.id.btn_shutter:
                SimpleDateFormat date = new SimpleDateFormat("yyyy_MM_dd_HH_mm_ss");
                synchronized (this) {
                    savedImagePath = Utils.getDCIMDirectory() + File.separator + date.format(new Date()).toString() + ".png";
                }
                Intent intent = new Intent(MainActivity.this, TableActivity.class);
                if (ocr_res != null) {
                    intent.putExtra("list", ocr_res);
                }
                else {
                    intent.putExtra("list", new ArrayList<String>().add("No Items Found"));
                }
                startActivity(intent);
                //Toast.makeText(MainActivity.this, "Save snapshot to " + savedImagePath, Toast.LENGTH_SHORT).show();
                break;
        }
    }

    @Override
    public boolean onTextureChanged(int inTextureId, int outTextureId, int textureWidth, int textureHeight) {
        String savedImagePath = "";
        synchronized (this) {
            savedImagePath = MainActivity.this.savedImagePath;
        }
        savedImagePath = new File(this.getExternalFilesDir(null), savedImagePath).getAbsolutePath();
        final boolean modified = predictor.process(inTextureId, outTextureId, textureWidth, textureHeight, savedImagePath);
        if (!savedImagePath.isEmpty()) {
            synchronized (this) {
                MainActivity.this.savedImagePath = "";
            }
        }

        lastFrameIndex++;
        if (lastFrameIndex >= 30) {
            final int fps = (int) (lastFrameIndex * 1e9 / (System.nanoTime() - lastFrameTime));
            runOnUiThread(new Runnable() {
                public void run() {
                    tvStatus.setText(Integer.toString(fps) + "fps");
                }
            });
            lastFrameIndex = 0;
            lastFrameTime = System.nanoTime();
        }

        String[] fetched = predictor.OCRResults();
        if (fetched == null) {
            return modified;
        }
        if (ocr_res == null) {
            ocr_res = new ArrayList<>();
            int i=0;
            while (fetched[i].equals("Unknown")) {
                if (i<fetched.length - 1) i++;
                else break;
            }
            ocr_res.add(fetched[i]);

            for (String s : fetched) {
                if (!ocr_res.contains(s) &&
                        !Objects.equals(s, "Unknown")) {
                    ocr_res.add(s);
                }
            }
        }
        else {
            for (String s : fetched) {
                if (!ocr_res.contains(s) &&
                        !Objects.equals(s, "Unknown")) {
                    ocr_res.add(s);
                }
            }
        }
        return modified;
    }

    @Override
    protected void onResume() {
        super.onResume();
        // Reload settings and re-initialize the predictor
        checkRun();
        // Open camera until the permissions have been granted
        if (!checkAllPermissions()) {
            svPreview.disableCamera();
        }
        svPreview.onResume();
    }

    @Override
    protected void onPause() {
        super.onPause();
        svPreview.onPause();
    }

    @Override
    protected void onDestroy() {
        if (predictor != null) {
            predictor.release();
        }
        super.onDestroy();
    }

    public void initView() {
        svPreview = (CameraSurfaceView) findViewById(R.id.sv_preview);
        svPreview.setOnTextureChangedListener(this);
        tvStatus = (TextView) findViewById(R.id.tv_status);
        btnSwitch = (ImageButton) findViewById(R.id.btn_switch);
        btnSwitch.setOnClickListener(this);
        btnShutter = (ImageButton) findViewById(R.id.btn_shutter);
        btnShutter.setOnClickListener(this);
    }

    @SuppressLint("SetTextI18n")
    public void checkRun() {
            try {

                Utils.copyAssets(this, labelPath);
                String labelRealDir = new File(
                        this.getExternalFilesDir(null),
                        labelPath).getAbsolutePath();

                Utils.copyAssets(this, configPath);
                String configRealDir = new File(
                        this.getExternalFilesDir(null),
                        configPath).getAbsolutePath();

                Utils.copyAssets(this, detModelPath);
                String detRealModelDir = new File(
                        this.getExternalFilesDir(null),
                        detModelPath).getAbsolutePath();

                Utils.copyAssets(this, classModelPath);
                String classRealModelDir = new File(
                        this.getExternalFilesDir(null),
                        classModelPath).getAbsolutePath();

                boolean ret = predictor.init(
                            this,
                            detRealModelDir,
                            classRealModelDir,
                            configRealDir,
                            labelRealDir,
                            cpuThreadNum,
                            cpuPowerMode);
                //tvStatus.setText(Boolean.toString(ret));

            } catch (Throwable e) {
                //tvStatus.setText("failure");
                e.printStackTrace();

            }
        }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
            if (grantResults[0] != PackageManager.PERMISSION_GRANTED || grantResults[1] != PackageManager.PERMISSION_GRANTED) {
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle("Permission denied")
                        .setMessage("Click to force quit the app, then open Settings->Apps & notifications->Target " +
                                "App->Permissions to grant all of the permissions.")
                        .setCancelable(false)
                        .setPositiveButton("Exit", new DialogInterface.OnClickListener() {
                            @Override
                            public void onClick(DialogInterface dialog, int which) {
                                MainActivity.this.finish();
                            }
                        }).show();
            }
    }

    private void requestAllPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE,
                Manifest.permission.CAMERA}, 0);

    }


    private boolean checkAllPermissions() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED
                && ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }
}
