import os
import time

import argparse
import tensorflow as tf
import tflite_runtime.interpreter as tflite
from tensorflow.python.keras import backend as K
import numpy as np
from PIL import Image

from DecodeUtil import non_max_suppression, process_mask
from ModelArgs import ModelArgs

def main(args):
    if args.model.lower() == "yolo480":
        modelPath = "Yolo/best_saved_model480/best_float32.tflite"
    elif args.model.lower() == "deeplab":
        modelPath = "DeepLab/deeplabv3_257_mv_gpu.tflite"
    elif args.model.lower() == "yolo288":
        modelPath = "Yolo/best_saved_model288/best_float32.tflite"
    else:
        print("The model given is not valid. Type either 'yolo' or 'deeplab'")
        return

    # Initialize interpreter
    inter = tflite.Interpreter(model_path=modelPath)
    inter.allocate_tensors()

    # Extract tensor details
    inputTens = inter.get_input_details()
    outputTens = inter.get_output_details()

    # Create input data
    inputList = CreateDataList(args.model.lower())

    # Warmup model (prøver uden først)
    inputShape = inputTens[0]["shape"]
    randomInputData = np.array(np.random.random_sample(inputShape), dtype=np.float32)
    inter.set_tensor(inputTens[0]["index"], randomInputData)
    inter.invoke()

    # Inference test
    x = 0

    with open(f"infTestRes{args.model.upper()}.csv", "w") as file:
        totalInfTime = 0
        totalPPTime = 0
        
        file.write("Index,InferTime,PostProcessTime\n")
        for input in inputList:            
            print(f"Running inference test for element {x} out of {len(inputList)-1}")

            inter.set_tensor(inputTens[0]["index"], input[0])
            infT1 = time.time()
            inter.invoke()
            infT2 = time.time()

            totalInfTime += infT2-infT1
            
            modelOut = inter.get_tensor(outputTens[0]["index"])

            if (args.model.lower() == "yolo480") or (args.model.lower() == "yolo288"):
                ppT1 = time.time()
                modelOutTens = tf.convert_to_tensor(modelOut)
                extraYoloData = tf.convert_to_tensor(inter.get_tensor(outputTens[1]["index"]))
                mask = DecodeYolo(modelOutTens, extraYoloData, input[0])
                ppT2 = time.time()
            elif args.model.lower() == "deeplab":
                ppT1 = time.time()
                mask = DecodeDeepLab(modelOut)
                ppT2 = time.time()
            else:
                print("Error! Model not found during inference test")

            totalPPTime += ppT2-ppT1

            file.write(f"{x},{'{:.10f}'.format(infT2-infT1)},{'{:.10f}'.format(ppT2-ppT1)}\n")
            # file.write(f"{x}: {t2-t1}\n")
            x += 1
        
        file.write(f"\nAvg Inference Time: {(totalInfTime/len(inputList))*1000} ms")
        file.write(f"\nAvg Post-Processing Time: {(totalPPTime/len(inputList))*1000} ms")
        file.write(f"\nAvg Total Time: {((totalInfTime + totalPPTime)/len(inputList))*1000} ms")
    
    x = 0
    # Accuracy test
    with open(f"accTestRes{args.model.upper()}.csv", "w") as file:
        accAccum = 0
        falseNegAccum = 0
        falsePosAccum = 0
        imgSize = 0
        file.write("Index,FullAcc,FalseNeg,FalsePos\n")
        for input in inputList:
            print(f"Running accuracy test for element {x} out of {len(inputList)-1}")

            inter.set_tensor(inputTens[0]["index"], input[0])
            inter.invoke()
            modelOut = inter.get_tensor(outputTens[0]["index"])

            if (args.model.lower() == "yolo480") or (args.model.lower() == "yolo288"):
                modelOutTens = tf.convert_to_tensor(modelOut)
                extraYoloData = tf.convert_to_tensor(inter.get_tensor(outputTens[1]["index"]))
                mask = DecodeYolo(modelOutTens, extraYoloData, input[0])
                imgSize = input[0].shape[1]*input[0].shape[2]
            elif args.model.lower() == "deeplab":
                mask = DecodeDeepLab(modelOut)
                imgSize = 257*257
            else:
                print("Error! Model not found during accuracy test")
            
            # keys = [False, True]

            # accDict = dict.fromkeys(keys, 0)
            # falsePosDict = dict.fromkeys(keys, 0)
            # predDict = dict.fromkeys(keys, 0)
            # falseNegDict = dict.fromkeys(keys, 0)
            # valDict = dict.fromkeys(keys, 0)

            accMask = np.logical_xor(mask, input[1])
            accMask = np.logical_not(accMask)
            accUni, accCount = np.unique(accMask, return_counts=True)
            accDict = dict(zip(accUni, accCount))

            invVal = np.logical_not(input[1])
            falsePosMask = np.logical_and(invVal, mask)
            falsePosUni, falsePosCount = np.unique(falsePosMask, return_counts=True)
            falsePosDict = dict(zip(falsePosUni, falsePosCount))
            predUni, predCount = np.unique(mask, return_counts=True)
            predDict = dict(zip(predUni, predCount))

            invPred = np.logical_not(mask)
            falseNegMask = np.logical_and(invPred, input[1])
            falseNegUni, falseNegCount = np.unique(falseNegMask, return_counts=True)
            falseNegDict = dict(zip(falseNegUni, falseNegCount))
            valUni, valCount = np.unique(input[1], return_counts=True)
            valDict = dict(zip(valUni, valCount))
            
            if True in accDict:
                acc = accDict[True]/imgSize
            else:
                acc = "DIVIDE_BY_0_ERROR"

            if True in falsePosDict:
                falsePos = falsePosDict[True]/predDict[True]
            else:
                falsePos = 0
            
            if True in falseNegDict:
                falseNeg = falseNegDict[True]/valDict[False]
            else:
                falseNeg = 0
            
            file.write(f"{x},{'{:.10f}'.format(acc)},{'{:.10f}'.format(falseNeg)},{'{:.10f}'.format(falsePos)}\n")

            accAccum += acc
            falseNegAccum += falseNeg
            falsePosAccum += falsePos
            x += 1
        file.write(f"\nAvg accuracy percent: {(accAccum/len(inputList))*100}%")
        file.write(f"\nAvg false negative percent: {(falseNegAccum/len(inputList)*100)}%")
        file.write(f"\nAvg false positive percent: {(falsePosAccum/len(inputList)*100)}%")


def CreateDataList(model):
    if model == "yolo480":
        dataPath = "Data_Yolo"
        maskPath = "Masks_Yolo"
    elif model == "deeplab":
        dataPath = "Data_DeepLab"
        maskPath = "Masks_DeepLab"
    elif model == "yolo288":
        dataPath = "Data_YoloSmall"
        maskPath = "Masks_YoloSmall"
    else:
        print("Model not recognised while creating input data")
        return None
    dataList = []
    dataDirList= []
    maskDirList = []

    dataFileList = list(sorted(os.listdir(dataPath)))
    maskFileList = list(sorted(os.listdir(maskPath)))

    for file in dataFileList:
        dataDirList.append(f"{dataPath}/{file}")

    for file in maskFileList:
        maskDirList.append(f"{maskPath}/{file}")

    for dir in dataDirList:
        currMask = ""

        for mask in maskDirList:
            if mask.split("_")[-1] == dir.split("_")[-1]:
                currMask = mask
                break
        
        pilMask = Image.open(currMask)
        pilImg = Image.open(dir)
        npImg = np.asarray(pilImg, dtype=np.float32)
        npMask = np.asarray(pilMask, dtype=bool)
        npMask = np.squeeze(npMask)
        npImg = (npImg-np.min(npImg))/(np.max(npImg)-np.min(npImg)) # Normalise npImg values from between 0 and 255 to between 0 and 1
        npImg = np.expand_dims(npImg, axis=0)
        dataList.append((npImg, npMask))
        # dataList.append((dir, currMask))
    
    return dataList

def DecodeDeepLab(modelOut):
    mask = modelOut[:, :, :, 0]

    mask = tf.squeeze(mask, axis=0)
    mask = K.sigmoid(mask)
    mask = tf.math.greater(mask, 0.9999999)

    maskNp = mask.numpy()

    return np.logical_not(maskNp)

def DecodeYolo(modelOutTens, extraYoloData, inputData):

    args = ModelArgs()
    p = non_max_suppression(modelOutTens,
                            args.conf,
                            args.iou,
                            agnostic=args.agnostic_nms,
                            max_det=args.max_det,
                            nc=args.nc,
                            classes=args.classes)

    proto = extraYoloData[-1] if len(extraYoloData) == 3 else extraYoloData

    proto = tf.transpose(proto, perm=[0, 3, 1, 2])

    for i, pred in enumerate(p):
        maskPred = pred[:, 6:]
        boxPred = pred[:, :4]

        if maskPred.shape[0] == 0:
            masks = tf.fill([1, inputData.shape[1], inputData.shape[2]], False)
        else:
           masks = process_mask(proto[i], maskPred, boxPred, inputData.shape[1:3], upsample=True)

    return masks[0].numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    args = parser.parse_args()
    main(args)