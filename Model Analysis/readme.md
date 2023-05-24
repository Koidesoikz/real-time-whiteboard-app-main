## Dependencies
* tensorflow
* tflite
* numpy
* pillow (PIL)
* torch
* torchvision

## Analysing the model
To run the analysis, simply run the script called `RunTest.py`

The script has one required argument, which is `--model` which should be either `yolo480`, `yolo288` or `deeplab`, depending on what model the script should analyze.

### Important before running
Make sure you extract the data from `Data_and_Masks.rar` into the folder.