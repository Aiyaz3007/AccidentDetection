TRAIN_ANNOTATIONS_FILE = r"dataset/train.json"
VAL_ANNOTATIONS_FILE = r"dataset/val.json"
IMAGES = r"dataset/images"

EPOCHS = 40
BATCH_SIZE = 8
DEVICE = "cpu" #"cuda","cpu"

RESIZE_FACTOR = (600,600)


INPUTDATATYPE = "video" #image,video,videos



MODEL_PATH = r"models/model_39_loss_0.045951_val_loss_0.155342.torch"
IMAGE_PATH = [r"accident_1.jpg","output.jpg"]
VIDEO_PATH = [r"accident_test_videos/1.mp4","output.mp4"]
FOLDER_OF_VIDEO = [r"accident_test_videos","output"]

CONFIDENCE_THRESHOLD = 0.6


classes = [
  "__background__",
  "Accident",
  "Car",
  "Bike"
]