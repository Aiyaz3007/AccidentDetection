## Accident Detection & Alert System

## Install & Dependencies
- python
- pytorch
- numpy
- opencv

  ## Installation

  ```
  pip install -r requirements.txt
  ```

## Use
- for train
  ```
  python train.py
  ```
- for test
  ```
  python test.py
  ```


## Directory Hierarchy
```
|—— .gitignore
|—— dataset
|    |—— annotations
|        |—— instances_default.json
|    |—— images
|        |—— accident_1.jpg
|        |—— accident_10.jpg
|        .
|        .
|        .
|    |—— info.txt
|    |—— train.json
|    |—— val.json
|—— dataset.py
|—— models
|    |—— loss.json
|    |—— model_22_loss_0.072028_val_loss_0.147941.torch
|    |—— model_39_loss_0.045951_val_loss_0.155342.torch
|—— requirements.txt
|—— src
|    |—— constants.py
|    |—— loss_curve.py
|    |—— test_utils.py
|    |—— train_utils.py
|    |—— utils.py
|    |—— __init__.py
|—— test.py
|—— train.py
```
