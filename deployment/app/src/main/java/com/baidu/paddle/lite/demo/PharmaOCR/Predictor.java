package com.baidu.paddle.lite.demo.PharmaOCR;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ColorMatrix;
import android.graphics.ColorMatrixColorFilter;
import android.graphics.Paint;
import android.graphics.RectF;
import android.util.Log;

import com.baidu.paddle.lite.MobileConfig;
import com.baidu.paddle.lite.PaddlePredictor;
import com.baidu.paddle.lite.PowerMode;
import com.baidu.paddle.lite.Tensor;

import java.io.File;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.Vector;
import static android.graphics.Color.*;
public class Predictor {
    private static final String TAG = "OCR_Predictor";

    // Detection model components
    private PaddlePredictor paddlePredictorDet = null;
    private int detInputSize = 640;
    //private float[] detMean = new float[]{0.485f, 0.456f, 0.406f};
    //private float[] detStd = new float[]{0.229f, 0.224f, 0.225f};

    private float[] detMean = new float[]{0.5f, 0.5f, 0.5f};
    private float[] detStd = new float[]{0.5f, 0.5f, 0.5f};


    // Classification model components
    private PaddlePredictor paddlePredictorCls = null;
    private int clsInputWidth = 64;
    private int clsInputHeight = 32;
    private float[] clsMean = new float[]{0.5f, 0.5f, 0.5f};
    private float[] clsStd = new float[]{0.5f, 0.5f, 0.5f};

    // Common components
    private boolean isLoaded = false;
    private float inferenceTime = 0;
    private Vector<String> wordLabels = new Vector<>();
    private Bitmap inputImage = null;
    private Bitmap outputImage = null;
    public String outputResult = "";
    private List<OCRResult> ocrResults = new ArrayList<>();
    private float scoreThreshold = 0.3f;


    public boolean init(Context appCtx, String modelDir, String labelPath,
                        int cpuThreadNum, String cpuPowerMode) {
        try {
            // Initialize detection model
            loadModel(appCtx, modelDir, cpuThreadNum, cpuPowerMode);

            // Load labels
            if (!loadLabel(appCtx, labelPath)) {
                Log.e(TAG, "Failed to load labels");
                return false;
            }

            isLoaded = (paddlePredictorDet != null) && (paddlePredictorCls != null);
            return isLoaded;

        } catch (Exception e) {
            Log.e(TAG, "Model initialization failed: " + e.getMessage());
            return false;
        }
    }

    protected void loadModel(Context appCtx, String modelPath, int cpuThreadNum, String cpuPowerMode) {
        // Release model if exists
        releaseModel();

        // Load model
        if (modelPath.isEmpty()) {
            return;
        }
        String realPath = modelPath;
        if (!modelPath.substring(0, 1).equals("/")) {
            // Read model files from custom path if the first character of mode path is '/'
            // otherwise copy model to cache from assets
            realPath = appCtx.getCacheDir() + "/" + modelPath;
            Utils.copyDirectoryFromAssets(appCtx, modelPath, realPath);
        }
        if (realPath.isEmpty()) {
            return;
        }
        MobileConfig config = new MobileConfig();
        config.setModelFromFile(realPath + File.separator + "detection_lite.nb");
        config.setThreads(cpuThreadNum);
        setPowerMode(config, cpuPowerMode);
        paddlePredictorDet = PaddlePredictor.createPaddlePredictor(config);

        MobileConfig configCls = new MobileConfig();
        configCls.setModelFromFile(realPath + File.separator + "text_classifier.nb");
        configCls.setThreads(cpuThreadNum);
        setPowerMode(configCls, cpuPowerMode);
        paddlePredictorCls = PaddlePredictor.createPaddlePredictor(configCls);
    }

    private void setPowerMode(MobileConfig config, String cpuPowerMode) {
        if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_LOW);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_FULL")) {
            config.setPowerMode(PowerMode.LITE_POWER_FULL);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_NO_BIND")) {
            config.setPowerMode(PowerMode.LITE_POWER_NO_BIND);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_HIGH")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_HIGH);
        } else if (cpuPowerMode.equalsIgnoreCase("LITE_POWER_RAND_LOW")) {
            config.setPowerMode(PowerMode.LITE_POWER_RAND_LOW);
        } else {
            Log.e(TAG, "unknown cpu power mode!");
        }
    }

    private boolean loadLabel(Context appCtx, String labelPath) {
        wordLabels.clear();
        try {
            InputStream assetsInputStream = appCtx.getAssets().open(labelPath);
            int available = assetsInputStream.available();
            byte[] lines = new byte[available];
            assetsInputStream.read(lines);
            assetsInputStream.close();
            String words = new String(lines);
            String[] contents = words.split("\n");
            for (String content : contents) {
                wordLabels.add(content.trim());
            }
            Log.i(TAG, "Label size: " + wordLabels.size());
            return true;
        } catch (Exception e) {
            Log.e(TAG, "Label loading error: " + e.getMessage());
            return false;
        }
    }
    public boolean runModel() {
        if (inputImage == null || !isLoaded) {
            return false;
        }

        Date startTime = new Date();

         //1. Run text detection
        List<RectF> textBoxes = runTextDetection(inputImage);
         //2. Run classification for each detected region
        ocrResults.clear();
        outputResult = "";

        for (RectF box : textBoxes) {
            Bitmap cropped = cropAndProcess(inputImage, box);
            String text = runTextClassification(cropped);
            ocrResults.add(new OCRResult(text, box));
            outputResult += text + "\n";
        }

        // 3. Create output image with bounding boxes
        outputImage = drawDetectionResults(inputImage, textBoxes);

        inferenceTime = (float) (new Date().getTime() - startTime.getTime());
        return true;
    }

    private List<RectF> runTextDetection(Bitmap image) {
        // 1. Preprocess image for detection
        Bitmap scaledImage = Bitmap.createScaledBitmap(image, detInputSize, detInputSize, true);
        float[] inputData = preprocessDetImage(scaledImage);

        // 2. Run detection model
        Tensor inputTensor = paddlePredictorDet.getInput(0);
        inputTensor.resize(new long[]{1, 3, detInputSize, detInputSize});
        inputTensor.setData(inputData);
        paddlePredictorDet.run();

        // 3. Postprocess results
        Tensor outputTensor = paddlePredictorDet.getOutput(0);
        return postprocessDetResults(outputTensor, image.getWidth(), image.getHeight());
    }

    private float[] preprocessDetImage(Bitmap image) {
        int channels = 3;
        int width = image.getWidth();
        int height = image.getHeight();
        float[] inputData = new float[channels * width * height];

        int[] pixels = new int[width * height];
        image.getPixels(pixels, 0, width, 0, 0, width, height);

        for (int x = 0; x < height; x++) {
            for (int y = 0; y < width; y++) {
                int color = pixels[y * width + x];
                float r = Color.red(color) / 255.0f;
                float g = Color.green(color) / 255.0f;
                float b = Color.blue(color) / 255.0f;

                // Normalize and convert to CHW format
                inputData[y * width + x] = (r - detMean[0]) / detStd[0];
                inputData[width * height + y * width + x] = (g - detMean[1]) / detStd[1];
                inputData[2 * width * height + y * width + x] = (b - detMean[2]) / detStd[2];
            }
        }
        return inputData;
    }

    private List<RectF> postprocessDetResults(Tensor outputTensor, int imgWidth, int imgHeight) {
        List<RectF> textBoxes = new ArrayList<>();
        float[] outputData = outputTensor.getFloatData();
        long[] outputShape = outputTensor.shape();

        // Output format: [batch_size, num_boxes, 6] where last dim is:
        // [label_id, confidence, x1, y1, x2, y2]
        int numBoxes = (int) outputShape[1];

        for (int i = 0; i < numBoxes; i++) {
            int offset = i * 4;
            //float confidence = outputData[offset + 1];

            //if (confidence > scoreThreshold) {
                float x1 = outputData[offset + 0];
                float y1 = outputData[offset + 1];
                float x2 = outputData[offset + 2];
                float y2 = outputData[offset + 3];

                // Convert from normalized [0,1] to image coordinates
                textBoxes.add(new RectF(
                        x1 * imgWidth,
                        y1 * imgHeight,
                        x2 * imgWidth,
                        y2 * imgHeight
                ));
            //}
        }
        return textBoxes;
    }

    private String runTextClassification(Bitmap croppedImage) {
        // 1. Preprocess for classification
        Bitmap processed = preprocessClsImage(croppedImage);
        float[] inputData = convertBitmapToFloatArray(processed, 1, "");

        // 2. Run classification
        Tensor inputTensor = paddlePredictorCls.getInput(0);
        inputTensor.resize(new long[]{1, 1, clsInputHeight, clsInputWidth});
        inputTensor.setData(inputData);
        paddlePredictorCls.run();

        // 3. Get classification result
        Tensor outputTensor = paddlePredictorCls.getOutput(0);
        return getTopClass(outputTensor.getFloatData());
    }

    private Bitmap preprocessClsImage(Bitmap image) {
        // 1. Convert to grayscale
        Bitmap grayscale = toGrayscale(image);

        // 2. Apply thresholding
        //Bitmap binary = applySimpleThreshold(grayscale);

        // 3. Resize to model input size
        return Bitmap.createScaledBitmap(grayscale, clsInputWidth, clsInputHeight, true);
    }

    private Bitmap toGrayscale(Bitmap original) {
        Bitmap grayscale = Bitmap.createBitmap(
                original.getWidth(),
                original.getHeight(),
                Bitmap.Config.ARGB_8888
        );

        Canvas canvas = new Canvas(grayscale);
        Paint paint = new Paint();
        ColorMatrix cm = new ColorMatrix();
        cm.setSaturation(0);
        paint.setColorFilter(new ColorMatrixColorFilter(cm));
        canvas.drawBitmap(original, 0, 0, paint);
        return grayscale;
    }

    private Bitmap applySimpleThreshold(Bitmap grayscale) {
        Bitmap binary = grayscale.copy(Bitmap.Config.ARGB_8888, true);
        int width = binary.getWidth();
        int height = binary.getHeight();
        int[] pixels = new int[width * height];
        binary.getPixels(pixels, 0, width, 0, 0, width, height);

        final int threshold = 128;
        for (int i = 0; i < pixels.length; i++) {
            int color = pixels[i];
            int r = Color.red(color);
            int g = Color.green(color);
            int b = Color.blue(color);
            int gray = (r + g + b) / 3;
            pixels[i] = gray > threshold ? Color.WHITE : Color.BLACK;
        }

        binary.setPixels(pixels, 0, width, 0, 0, width, height);
        return binary;
    }

    private float[] convertBitmapToFloatArray(Bitmap image, int channels, String inputColorFormat) {
        float[] inputMean = clsMean;
        float[] inputStd = clsStd;
        int width = image.getWidth();
        int height = image.getHeight();
        float[] inputData = new float[channels * width * height];
        if (channels == 3) {
            int[] channelIdx = null;
            if (inputColorFormat.equalsIgnoreCase("RGB")) {
                channelIdx = new int[]{0, 1, 2};
            } else if (inputColorFormat.equalsIgnoreCase("BGR")) {
                channelIdx = new int[]{2, 1, 0};
            } else {
                Log.i(TAG, "Unknown color format " + inputColorFormat + ", only RGB and BGR color format is " +
                        "supported!");
                return null;
            }
            int[] channelStride = new int[]{width * height, width * height * 2};
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float[] rgb = new float[]{(float) red(color) / 255.0f, (float) green(color) / 255.0f,
                            (float) blue(color) / 255.0f};
                    inputData[y * width + x] = (rgb[channelIdx[0]] - inputMean[0]) / inputStd[0];
                    inputData[y * width + x + channelStride[0]] = (rgb[channelIdx[1]] - inputMean[1]) / inputStd[1];
                    inputData[y * width + x + channelStride[1]] = (rgb[channelIdx[2]] - inputMean[2]) / inputStd[2];
                }
            }
        } else if (channels == 1) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    int color = inputImage.getPixel(x, y);
                    float gray = (float) (red(color) + green(color) + blue(color)) / 3.0f / 255.0f;
                    inputData[y * width + x] = (gray - inputMean[0]) / inputStd[0];
                }
            }
        } else {
            Log.i(TAG, "Unsupported channel size " + Integer.toString(channels) + ",  only channel 1 and 3 is " +
                    "supported!");
            return null;
        }
        return inputData;
    }

    private String getTopClass(float[] scores) {
        int maxIndex = 0;
        float maxScore = -Float.MAX_VALUE;
        for (int i = 0; i < scores.length; i++) {
            if (scores[i] > maxScore) {
                maxScore = scores[i];
                maxIndex = i;
            }
        }
        return wordLabels.size() > maxIndex ? wordLabels.get(maxIndex) : "Unknown";
    }

    private Bitmap cropAndProcess(Bitmap original, RectF box) {
        try {
            int left = Math.max(0, (int) box.left);
            int top = Math.max(0, (int) box.top);
            int right = Math.min(original.getWidth(), (int) box.right);
            int bottom = Math.min(original.getHeight(), (int) box.bottom);

            return Bitmap.createBitmap(original, left, top, right - left, bottom - top);
        } catch (Exception e) {
            Log.e(TAG, "Cropping error: " + e.getMessage());
            return original;
        }
    }

    private Bitmap drawDetectionResults(Bitmap original, List<RectF> boxes) {
        Bitmap result = original.copy(Bitmap.Config.ARGB_8888, true);
        Canvas canvas = new Canvas(result);
        Paint paint = new Paint();
        paint.setStyle(Paint.Style.STROKE);
        paint.setStrokeWidth(2.0f);

        int[] colors = {Color.RED, Color.BLUE, Color.GREEN, Color.YELLOW, Color.CYAN};

        for (int i = 0; i < boxes.size(); i++) {
            paint.setColor(colors[i % colors.length]);
            canvas.drawRect(boxes.get(i), paint);
        }

        return result;
    }

    public void setInputImage(Bitmap image) {
        if (image != null) {
            this.inputImage = image.copy(Bitmap.Config.ARGB_8888, true);
        }
    }

    public void releaseModel() {
        paddlePredictorDet = null;
        paddlePredictorCls = null;
        isLoaded = false;
    }

    public boolean isLoaded() {
        return isLoaded;
    }

    public float inferenceTime() {
        return inferenceTime;
    }

    public Bitmap outputImage() {
        return outputImage;
    }

    public String outputResult() {
        return outputResult;
    }

    public List<OCRResult> getOCRResults() {
        return ocrResults;
    }

    public static class OCRResult {
        public final String text;
        public final RectF box;

        public OCRResult(String text, RectF box) {
            this.text = text;
            this.box = box;
        }
    }
}