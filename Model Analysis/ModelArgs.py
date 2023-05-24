class ModelArgs:
    def __init__(self):
        self.Test()

    def Test(self):
        with open("args.txt", "r") as file:
            lines = file.readlines()
            for line in lines:
                key, value = line.strip().split(":")
                
                if key == "conf":
                    if value == "null":
                        self.conf = 0.25
                    else:
                        self.conf = float(value)
                elif key == "iou":
                    self.iou = float(value)
                elif key == "agnostic_nms":
                    self.agnostic_nms = bool(value)
                elif key == "max_det":
                    self.max_det = int(value)
                elif key == "nc":
                    self.nc = int(value)
                elif key == "classes":
                    if value == "none":
                        self.classes = None
                    else:
                        self.classes = value
                else:
                    print("Error! Unknown key detected in args.txt!")