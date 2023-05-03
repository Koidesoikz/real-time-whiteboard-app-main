package com.whiteboardapp.core.pipeline;


import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.os.SystemClock;
import android.util.Log;

import com.whiteboardapp.common.AppUtils;
import com.whiteboardapp.common.CustomLogger;
import com.whiteboardapp.common.DebugTags;
import com.whiteboardapp.common.ModelArgs;

import org.checkerframework.checker.formatter.FormatUtil;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.tensorflow.Graph;
import org.tensorflow.Operand;
import org.tensorflow.Output;
import org.tensorflow.op.Ops;
import org.tensorflow.op.Scope;
import org.tensorflow.op.core.Constant;
import org.tensorflow.op.linalg.Transpose;
import org.tensorflow.lite.Tensor;
import org.tensorflow.lite.InterpreterFactory;
import org.tensorflow.lite.InterpreterApi;
import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.task.vision.segmenter.ColoredLabel;
import org.tensorflow.lite.task.vision.segmenter.Segmentation;
import org.tensorflow.lite.task.vision.segmenter.ImageSegmenter;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt32;
import org.tensorflow.types.TUint8;

import java.security.InvalidKeyException;
import java.nio.MappedByteBuffer;
import java.util.ArrayList;
import java.util.List;

// Segmentation using DeepLab model.
public class Segmentator {
    public CustomLogger logger = new CustomLogger();
    public final String TAG = "SegmentationTask";
    private final int NUM_THREADS = 4;
    private final String SEGMENTATION_MODEL_NAME = "best_model.tflite";
    private final int ALPHA_VALUE = 128;

    private ImageSegmenter imageSegmenter;
    private TensorImage maskTensor;
    private int[] pixelsGlobal;
    private InterpreterApi tflite;
    private TensorBuffer buffer;

    public Segmentator(Context context) {
        buffer = TensorBuffer.createFixedSize(new int[]{1, 3, 480, 480}, DataType.FLOAT32);

        try {
            // DET HER ER MIG DER LIGE PRØVER NOGET!!!!!!
            /*ImageProcessor imageProcessor = new ImageProcessor.Builder().build();
            TensorImage tensorImage = new TensorImage(DataType.UINT8)*/
            MappedByteBuffer model = FileUtil.loadMappedFile(context, SEGMENTATION_MODEL_NAME);

            tflite = InterpreterApi.create(model, new InterpreterApi.Options());
            System.out.println("YOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO");
            // imageSegmenter = ImageSegmenter.createFromFile(context, SEGMENTATION_MODEL_NAME);
        } catch (Exception exception) {
            throw new RuntimeException("Exception occurred while loading model from file" + exception);
        }
    }

    // Performs segmentation of the given image. 
    // Returns a Mat representing the resulting segmentation map 
    public Mat segmentate(Bitmap image) {
        logger = new CustomLogger();
        float[] res;
        int[] shape;
        ArrayList<Tensor> resTensors = new ArrayList<Tensor>();
        //Graph g = new Graph();
        //Scope scope = g.baseScope();
        Ops ops = Ops.create();


        long startTime = System.currentTimeMillis();
        // Do



        TensorImage tensorImage = TensorImage.fromBitmap(image);
        TensorImage tensorImageFloat = TensorImage.createFrom(tensorImage, DataType.FLOAT32);
        logger.AddTime(System.currentTimeMillis() - startTime, "TensorImage");



        startTime = System.currentTimeMillis();
        /*
        System.out.println("TensorImage:");
        System.out.println(tensorImage.getBuffer().toString());
        System.out.println("TensorImageFloat:");
        System.out.println(tensorImageFloat.getBuffer().toString());
        System.out.println("Buffer:");
        System.out.println(buffer.getBuffer().toString());
        */


        if (tflite != null) {
            tflite.run(tensorImageFloat.getBuffer(), buffer.getBuffer());

            resTensors.add(tflite.getOutputTensor(0));
            Tensor tempTensor = tflite.getOutputTensor(1);
            int[] perm = new int[]{0, 3, 1, 2};

            // JEG BEGÅR SELVMORD :))))))))))))))))
            Constant<TInt32> c1 = ops.constant(perm);
            Constant<TUint8> c2 = ops.constant(tempTensor.asReadOnlyBuffer().array());
            //Tensor tensor = ops.linalg.transpose(tempTensor, ops.constant(perm));

            Output<TInt32> y = ops.linalg.transpose(c1, c2).y();
            //Output<TUint8> y = Transpose.create(scope, c2 ,c1).y();

            resTensors.add(tempTensor);
            res = buffer.getFloatArray();
            shape = buffer.getShape();
            System.out.println(shape);
            System.out.println(res[0]);
        }

        ModelArgs testArg = new ModelArgs();
        /*
        [0][0][0][0]
        [0][1][0][0]    [0][0][0][1]
        */

        //Tensor testTensor =

        //res.ToMask();

        //List<Segmentation> results = imageSegmenter.segment(tensorImage);
        logger.AddTime(System.currentTimeMillis() - startTime, "ImageSegmenter");

        startTime = System.currentTimeMillis();
        // Resize seg map to input image size.
        /*
        Bitmap maskBitmap = createMaskBitmapAndLabels(
                results.get(0), image.getWidth(),
                image.getHeight()
        ); */
        logger.AddTime(System.currentTimeMillis() - startTime, "ResizeSegMap");

        Bitmap maskBitmap = image;

        Mat returnMat = new Mat();
        Utils.bitmapToMat(maskBitmap, returnMat);


        startTime = System.currentTimeMillis();
        Mat imgSegMap = createImgSegMap(maskBitmap, image.getWidth(), image.getHeight());
        logger.AddTime(System.currentTimeMillis() - startTime, "CreateImgSegMap");
        logger.Log(DebugTags.SegmentorTag);
        return returnMat;
    }

    // Method converted from Tensorflow Kotlin tutorial:
    // https://github.com/tensorflow/examples/tree/master/lite/examples/image_segmentation/android/lib_task_api/src/main/java/org/tensorflow/lite/examples/imagesegmentation/tflite
    private Bitmap createMaskBitmapAndLabels(Segmentation segmentation, int width, int height) {
        List<ColoredLabel> coloredLabels = segmentation.getColoredLabels();
        ArrayList<Integer> colors = new ArrayList<>();
        for (ColoredLabel coloredLabel : coloredLabels) {
            int rgb = coloredLabel.getArgb();
            // Create color setting alpha value (transparency) to 128 as the mask will be displayed on top of original image.
            colors.add(Color.argb(ALPHA_VALUE, Color.red(rgb), Color.green(rgb), Color.blue(rgb)));
        }
        // Use completely transparent for the background color.
        colors.set(0, Color.TRANSPARENT);

        // Create the mask bitmap with colors and the set of detected labels.
        maskTensor = segmentation.getMasks().get(0);
        byte[] maskArray = maskTensor.getBuffer().array();
        int[] pixels = new int[maskArray.length];
        for (int i = 0; i < maskArray.length; i++) {
            int color = colors.get(maskArray[i]);
            pixels[i] = color;
        }

        // Scale the maskBitmap to the same size as the input image.
        Bitmap maskBitmap = Bitmap.createBitmap(
                pixels, maskTensor.getWidth(), maskTensor.getHeight(),
                Bitmap.Config.ARGB_8888
        );
        // XXX: Scaling her??
        return maskBitmap;

    }

    private Mat createImgSegMap(Bitmap maskBitmap, int width, int height) {

        // Create segmap
        Mat segMapBgr = new Mat();
        Utils.bitmapToMat(maskBitmap, segMapBgr);

        Mat segMapGrey = new Mat();
        Imgproc.cvtColor(segMapBgr, segMapGrey, Imgproc.COLOR_BGR2GRAY);

        Mat segMapBinary = new Mat(segMapGrey.rows(), segMapGrey.cols(), CvType.CV_8U);

        byte[] bufferSegGray = AppUtils.getBuffer(segMapGrey);
        byte[] bufferSegBinary = AppUtils.getBuffer(segMapBinary);

        for (int i = 0; i < bufferSegGray.length; i++) {
            if (bufferSegGray[i] == 68) {
                bufferSegBinary[i] = -1;
            } else {
                bufferSegBinary[i] = 0;

            }
        }
        segMapBinary.put(0, 0, bufferSegBinary);

        Imgproc.resize(segMapBinary, segMapBinary, new Size(width, height));
        Mat kernel = Imgproc.getStructuringElement(Imgproc.CV_SHAPE_ELLIPSE, new Size(20, 20));
        Mat imgSegMapDilated = new Mat();
        Imgproc.dilate(segMapBinary, imgSegMapDilated, kernel, new Point(-1, -1), 11);

        return imgSegMapDilated;
    }

}
