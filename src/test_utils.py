import os
from .utils import isColab
import cv2
from os.path import join,isfile
import torch
from src import constants

if isColab():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


def VideoPrediction(model,device,classes,videoFolder,outFolder):
    os.makedirs(outFolder,exist_ok=True)
    MainBar = tqdm(total=len(os.listdir(videoFolder)),desc="Video Done!")
    for fileName in os.listdir(videoFolder):

        
        # Load video file
        cap = cv2.VideoCapture(join(videoFolder,fileName))
        
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Bar = tqdm(total=length,desc=fileName)
        # Define the output video file
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        out = cv2.VideoWriter(join(outFolder,fileName), fourcc, 20.0, (600,600))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        model.eval()
        count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # print(f"{count}/{length}")
            height, width, channels = frame.shape
            img = torch.Tensor(cv2.resize(frame,constants.RESIZE_FACTOR)/255).permute(2,0,1)

            frame = torch.tensor(img*255, dtype=torch.uint8)

            with torch.no_grad():
                prediction = model([img.to(device)])
                pred = prediction[0]
                boxes = pred["boxes"].cpu().numpy()
                labels = pred["labels"].cpu()
                scores = pred["scores"].cpu()

            frame = frame.permute(1,2,0).numpy()
            for bbox,label,score in zip(boxes,labels,scores):
                if score > constants.CONFIDENCE_THRESHOLD:
                  xmin,ymin,xmax,ymax = list(map(int,(bbox)))
                  className = classes[label.item()]
                  if className == "Accident":
                    if round(score.item(),2)*100 >= 95:
                      continue
                  color1 = (0, 255, 255)
                  color2 = (10, 0, 255)

                  cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
                  cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

            # Write the processed frame to the output video
            out.write(frame)
            Bar.update(1)
        
        # Release video capture and writer objects
        cap.release()
        out.release()
        MainBar.update(1)
        

def singleVideoPrediction(model,device,classes:str,videoPath:str,outPath:str=""):
    assert isfile(videoPath) and videoPath.endswith("mp4"),"Video is not Supported, Check is that filename is proper, and File Must be mp4 format"
    assert outPath.endswith("mp4"),"Video is not Supported, Check is that filename is proper, and File Must be mp4 format"
    
    # Load video file
    cap = cv2.VideoCapture(videoPath)
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    Bar = tqdm(total=length,desc=videoPath.split("/")[-1])
    # Define the output video file
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(outPath, fourcc, 20.0, constants.RESIZE_FACTOR)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    model.eval()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = torch.Tensor(cv2.resize(frame,constants.RESIZE_FACTOR)/255).permute(2,0,1)

        frame = torch.tensor(img*255, dtype=torch.uint8)

        with torch.no_grad():
            prediction = model([img.to(device)])
            pred = prediction[0]
            boxes = pred["boxes"].cpu().numpy()
            labels = pred["labels"].cpu()
            scores = pred["scores"].cpu()

        frame = frame.permute(1,2,0).numpy()
        for bbox,label,score in zip(boxes,labels,scores):
            if score > constants.CONFIDENCE_THRESHOLD:
                xmin,ymin,xmax,ymax = list(map(int,(bbox)))
                className = classes[label.item()]
                if className == "Accident":
                    if round(score.item(),2)*100 >= 95:
                        continue
                color1 = (0, 255, 255)
                color2 = (10, 0, 255)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
                cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

        # Write the processed frame to the output video
        out.write(frame)
        Bar.update(1)
    
    # Release video capture and writer objects
    cap.release()
    out.release()
    

def singleImagePrediction(model,device,classes:str,imagePath:str,outPath:str=""):
    assert isfile(imagePath) and imagePath.endswith("jpg"),"Video is not Supported, Check is that filename is proper, and File Must be jpg format"
    assert outPath.endswith("jpg"),"Image is not Supported, Check is that filename is proper, and File Must be jpg format"
    
    # Load video file
    length = 1
    Bar = tqdm(total=length,desc=imagePath.split("/")[-1])
    imageData = cv2.imread(imagePath)
    img = torch.Tensor(cv2.resize(imageData,constants.RESIZE_FACTOR)/255).permute(2,0,1)

    frame = torch.tensor(img*255, dtype=torch.uint8)
    model.eval()
    with torch.no_grad():
        prediction = model([img.to(device)])
        pred = prediction[0]
        boxes = pred["boxes"].cpu().numpy()
        labels = pred["labels"].cpu()
        scores = pred["scores"].cpu()

    frame = frame.permute(1,2,0).numpy()
    for bbox,label,score in zip(boxes,labels,scores):
        if score > constants.CONFIDENCE_THRESHOLD:
            xmin,ymin,xmax,ymax = list(map(int,(bbox)))
            className = classes[label.item()]
            if className == "Accident":
                if round(score.item(),2)*100 >= 50:
                    continue
            color1 = (0, 255, 255)
            color2 = (10, 0, 255)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
            cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 2)

    Bar.update(1)
    if constants.DEVICE == "cpu":
        imageOut = frame
    else:
        imageOut = frame
    cv2.imwrite(outPath,frame)
    print(f"\nPrediction Saved : {outPath}")
    