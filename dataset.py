import os
import json
try:
    import kaggle 
except ModuleNotFoundError:
  print("kaggle not found! \nInstalling Kaggle")
  os.system("pip install kaggle")

os.system("kaggle datasets download -d aiyazm/accidentdetection")
