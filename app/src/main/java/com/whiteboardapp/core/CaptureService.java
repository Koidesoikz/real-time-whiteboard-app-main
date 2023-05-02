package com.whiteboardapp.core;


import android.content.Context;
import android.graphics.Bitmap;
import android.util.Log;

import com.whiteboardapp.common.CustomLogger;
import com.whiteboardapp.common.DebugTags;
import com.whiteboardapp.controller.MatConverter;
import com.whiteboardapp.core.pipeline.Binarization;
import com.whiteboardapp.core.pipeline.ChangeDetector;
import com.whiteboardapp.core.pipeline.Segmentator;

import org.checkerframework.checker.units.qual.C;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import com.whiteboardapp.common.AppUtils;

import static android.content.ContentValues.TAG;

import java.net.ServerSocket;

public class CaptureService {
    private final Context appContext;
    private final Mat currentModel;
    private final ChangeDetector changeDetector;

    public CaptureService(int defaultWidth, int defaultHeight, Context appContext) {
        this.appContext = appContext;
        currentModel = new Mat(defaultHeight / 2, defaultWidth / 2, CvType.CV_8UC1);
        currentModel.setTo(new Scalar(255));
        changeDetector = new ChangeDetector();
    }

    // Runs image through the image processing pipeline
    public Mat capture(Mat imgBgr) {
        CustomLogger logger = new CustomLogger();

        long startTime = System.currentTimeMillis();

        Mat dSized = new Mat();
        Imgproc.resize(imgBgr, dSized, new Size(480, 480), Imgproc.INTER_NEAREST);

        // Segmentation
        Mat matPerspectiveRgb = new Mat();
        Imgproc.cvtColor(dSized, matPerspectiveRgb, Imgproc.COLOR_BGR2RGB);
        Segmentator segmentator = new Segmentator(appContext);
        Bitmap bitmapRgb = MatConverter.matToBitmap(matPerspectiveRgb);
        System.out.println("Pre Segmentate");
        Mat imgSegMap = segmentator.segmentate(bitmapRgb);

        return imgSegMap;
        /*
        logger.AddTime(System.currentTimeMillis() - startTime, "Segmentation");
        startTime = System.currentTimeMillis();

        // Binarize a gray scale version of the image.
        Mat imgWarpGray = new Mat();
        Imgproc.cvtColor(dSized, imgWarpGray, Imgproc.COLOR_BGR2GRAY);
        Mat imgBinarized = Binarization.binarize(imgWarpGray);

        logger.AddTime(System.currentTimeMillis() - startTime,"Binarize");
        startTime = System.currentTimeMillis();

        // Remove segments before change detection.
        Mat currentModelCopy = removeSegmentArea(imgBinarized, currentModel, imgSegMap, dSized);

        logger.AddTime(System.currentTimeMillis() - startTime, "Remove segments");
        startTime = System.currentTimeMillis();

        // Change detection
        Mat imgPersistentChanges = changeDetector.detectChanges(imgBinarized, currentModelCopy);

        logger.AddTime(System.currentTimeMillis() - startTime, "Change detection");
        startTime = System.currentTimeMillis();

        // Update current model with persistent changes.
        updateModel(imgBinarized, imgPersistentChanges);

        logger.AddTime(System.currentTimeMillis() - startTime, "Update current model");
        logger.Log(DebugTags.ImageProcessingPipelineTag);

        return currentModel;

         */
    }

    // Removes segment area from image
    private Mat removeSegmentArea(Mat binarizedImg, Mat currentModel, Mat imgSegMap, Mat imgPerspective) {
        Mat currentModelCopy = new Mat();
        currentModel.copyTo(currentModelCopy);

//        for (int i = 0; i < binarizedImg.rows(); i++) {
//            for (int j = 0; j < binarizedImg.cols(); j++) {
//                if (imgSegMap.get(i, j)[0] == 255) {
//                    binarizedImg.put(i, j, 255);
//                    currentModelCopy.put(i, j, 255);
////                    imgPerspective.put(i, j, 255, 0, 0);
//                }
//            }
//        }

        byte[] bufferBinarized = AppUtils.getBuffer(binarizedImg);
        byte[] bufferSegmap = AppUtils.getBuffer(imgSegMap);
        byte[] bufferModel = AppUtils.getBuffer(currentModelCopy);

        for (int i = 0; i < bufferBinarized.length; i++) {
            if (bufferSegmap[i] == -1) {
                bufferBinarized[i] = -1;
                bufferModel[i] = -1;
            }
        }

        binarizedImg.put(0, 0, bufferBinarized);
        currentModelCopy.put(0, 0, bufferModel);

        return currentModelCopy;
    }

    private void updateModel(Mat imgBinarized, Mat imgPersistentChanges) {
        byte[] bufferModel = AppUtils.getBuffer(currentModel);
        byte[] bufferChanges = AppUtils.getBuffer(imgPersistentChanges);
        byte[] bufferBinarized = AppUtils.getBuffer(imgBinarized);

        for (int i = 0; i < bufferModel.length; i++) {
            if (bufferChanges[i] == -1) {
                bufferModel[i] = bufferBinarized[i];
            }
        }

//        for (int i = 0; i < currentModel.rows(); i++) {
//            for (int j = 0; j < currentModel.cols(); j++) {
//                if (imgPersistentChanges.get(i, j)[0] == 255) {
//                    currentModel.put(i, j, imgBinarized.get(i, j)[0]);
//                }
//            }
//        }

        currentModel.put(0, 0, bufferModel);
    }

}
