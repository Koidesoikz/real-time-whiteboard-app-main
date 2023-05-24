## Dependencies
* Ultralytics (to install: `pip install ultralytics`)

## Training the YOLO model
Use the `TrainWhiteboardSegmenter.py` script to train a model

Use the `Export.py` script to convert the trained model to a TFLite model

### Important before training
Extract the desired dataset in YOLO format, to the folder `datasets/whiteboard`

The final structure should look something like this:

```
datasets
    whiteboard
        images
            train
                (training imgs)
            val
                (validation imgs)
        labels
            train
                (training labels)
             val
                (validation labels)
```

### Important before converting .pt to .tflite
Make sure to change the `path` variable in `Export.py`, to the path of the model to convert.

**Note:** There might be problems with converting the model to a TFLite model on a windows machine. It should however work on a Linux machine, or in a WSL environment.