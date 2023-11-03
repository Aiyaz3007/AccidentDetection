from src import (singleImagePrediction,singleVideoPrediction,VideoPrediction,
                 constants)
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
from os.path import exists
import warnings
warnings.filterwarnings("ignore")

# classes
with open(constants.TRAIN_ANNOTATIONS_FILE,"r") as f:
  classes = [cat["name"] for cat in json.load(f)["categories"]]
  total_classes = classes.insert(0,"__background__")


# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))


device = torch.device(constants.DEVICE) if constants.DEVICE == "cuda" and  torch.cuda.is_available() else torch.device(constants.DEVICE)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True, weight_decay=1e-4)
print("Device:",device)

if constants.INPUTDATATYPE != "":
    if constants.INPUTDATATYPE == "image":
        if exists(constants.IMAGE_PATH[0]):
            singleImagePrediction(model=model,
                                device=device,
                                classes=classes,
                                imagePath=constants.IMAGE_PATH[0],
                                outPath=constants.IMAGE_PATH[1])
    if constants.INPUTDATATYPE == "video":
        if exists(constants.VIDEO_PATH[0]):
            singleVideoPrediction(model=model,
                                device=device,
                                classes=classes,
                                videoPath=constants.VIDEO_PATH[0],
                                outPath=constants.VIDEO_PATH[1])
    elif constants.INPUTDATATYPE == "videos":
        if exists(constants.FOLDER_OF_VIDEO[0]):
            VideoPrediction(model=model,
                            device=device,
                            classes=classes,
                            videoFolder=constants.FOLDER_OF_VIDEO,
                            outFolder=constants.FOLDER_OF_VIDEO[1])
else:
    print("\nNo Prediction Happened!")
    