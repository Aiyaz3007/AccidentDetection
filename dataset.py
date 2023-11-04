import os
import json
try:
    import kaggle 
except ModuleNotFoundError:
  print("kaggle not found! \nInstalling Kaggle")
  os.system("pip install kaggle")
  
import os
import json

kaggle_data = {"username":"aiyazm","key":"06b7b710ad258c1d23ccd666b1363bb2"}

os.makedirs(".kaggle",exist_ok=True)
with open(".kaggle/kaggle.json","w") as f:
  json.dump(kaggle_data,f)
  print("Done!")

os.environ["KAGGLE_CONFIG_DIR"] = ".kaggle/kaggle.json"

os.system("kaggle datasets download -d aiyazm/accidentdetection")
