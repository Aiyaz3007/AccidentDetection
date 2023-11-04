import os
from .utils import isColab
import cv2
from os.path import join,isfile
import torch
from src import constants
from torchvision import transforms

if isColab():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

transform = transforms.Compose([
    transforms.ToPILImage(),  # Convert to PIL Image
    transforms.Resize((600, 600)),  # Resize to 600x600
    transforms.ToTensor()  # Convert to PyTorch tensor
])

def VideoPrediction(model,device,classes,videoFolder,outFolder):
    os.makedirs(outFolder,exist_ok=True)
    MainBar = tqdm(total=len(os.listdir(videoFolder)),desc="Video Done!")
    for fileName in os.listdir(videoFolder):

        
        # Load video file
        cap = cv2.VideoCapture(join(videoFolder,fileName))
        
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        Bar = tqdm(total=length,desc=fileName)
        # Define the output video file
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(join(outFolder,fileName), fourcc, 20.0, (600,600))
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        model.eval()
        count = 1
        while True:
            ret, Mainframe = cap.read()
            if not ret:
                break
            # print(f"{count}/{length}")
            Mainframe = cv2.resize(Mainframe,constants.RESIZE_FACTOR)
            frame = Mainframe.copy()
            img = torch.Tensor(Mainframe/255).permute(2,0,1)

            # frame = torch.tensor(img*255, dtype=torch.uint8)

            with torch.no_grad():
                prediction = model([img.to(device)])

            # You can access the predictions, which include bounding boxes, labels, and scores
            boxes = prediction[0]['boxes']
            labels = prediction[0]['labels']
            scores = prediction[0]['scores']


            for bbox,label,score in zip(boxes,labels,scores):
                if score > constants.CONFIDENCE_THRESHOLD:
                    xmin,ymin,xmax,ymax = list(map(int,(bbox)))
                    className = classes[label.item()]
                    if className == "Accident":
                        if round(score.item(),2)*100 <= 95:
                            continue
                    color1 = (0, 255, 255)
                    color2 = (10, 0, 255)
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
                    cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color2, 1)

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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
                    if round(score.item(),2)*100 <= 95:
                        continue
                color1 = (0, 255, 255)
                color2 = (10, 0, 255)

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
                cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color2, 1)

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
    img = cv2.resize(imageData,constants.RESIZE_FACTOR)
    frame = img.copy()
    img = img/255
    img = torch.as_tensor(img, dtype=torch.float32)
    img = img.permute(2,0,1)
    
    with torch.no_grad():
        prediction = model([img.to(device)])

    # You can access the predictions, which include bounding boxes, labels, and scores
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']


    for bbox,label,score in zip(boxes,labels,scores):
        if score > constants.CONFIDENCE_THRESHOLD:
            xmin,ymin,xmax,ymax = list(map(int,(bbox)))
            className = classes[label.item()]
            if className == "Accident":
                if round(score.item(),2)*100 <= 95:
                    continue
            color1 = (0, 255, 255)
            color2 = (10, 0, 255)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color1, 2)
            cv2.putText(frame, f"{className} :{round(score.item(),2)*100}%", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color2, 1)
    Bar.update(1)
    cv2.imwrite(outPath,frame)
    print(f"\nPrediction Saved : {outPath}")
    