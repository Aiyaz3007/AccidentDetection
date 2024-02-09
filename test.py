from src import (singleImagePrediction,singleVideoPrediction,VideoPrediction,
                 constants)
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import json
from os.path import exists
import os
import warnings
warnings.filterwarnings("ignore")



classes = constants.classes
# model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))



device = torch.device(constants.DEVICE)


if constants.DEVICE == "cpu":
    state_dict = torch.load(constants.MODEL_PATH,map_location=torch.device(device))
    model.load_state_dict(state_dict)
else:
    # cuda 
    state_dict = torch.load(constants.MODEL_PATH)
    model.load_state_dict(state_dict)
    
model.to(device)
model.eval()

print("Device:",device)
os.makedirs("tmp", exist_ok=True)

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
                            videoFolder=constants.FOLDER_OF_VIDEO[0],
                            outFolder=constants.FOLDER_OF_VIDEO[1])
else:
    print("\nNo Prediction Happened!")
    