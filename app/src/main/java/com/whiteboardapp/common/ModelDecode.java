package com.whiteboardapp.common;

import org.tensorflow.lite.Tensor;

public final class ModelDecode {
    private static ModelArgs modelArgs;

    private ModelDecode(){
        modelArgs = new ModelArgs();
    }

    // Functions
    public static Tensor Decode(Tensor data){
        Tensor p = NonMaxSuppression(data);

        return null;
    }

    public static Tensor NonMaxSuppression(Tensor data){
        return null;
    }

    public static Tensor ProcessMask(Tensor proto, Tensor maskPred, Tensor boxPred, int[] shape, Boolean upsamle){

        return null;
    }

    // Funktioner jeg er ret sikker på ikke er nødvendige
    // xywh2xyxy
    // BoxIOU

}