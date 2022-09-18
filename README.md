# IoT-Activity-Detection-Clean
Updating

## Dataset Introduction
Currently, we are using a dataset consists of 951 videos.  
First, we use [Detectron2](https://github.com/facebookresearch/detectron2) to extract human body keypoints feature.
In each frame, there are 17 keypoints represented by their 2D coordinates (x,y). For each video,
the number of feature is 17 * 2 * n_frames.  
Second, we interpolate the feature sequence to length 100, then we have 17 * 2 * 100 = 3400 features for each video.  
The interpolated data can be found in the folder `feature_archive/all_feature_interp951.npz` or [download](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing)
here. 
```python
# How to use the dataset
import numpy as np

data_path = "feature_archive/all_feature_interp951.npz"
all_data = np.load(data_path, allow_pickle=True)
X_all = all_data["X"]   # shape(951, 3400)
Y_all = all_data["Y"]

X_all = X_all.reshape(-1, 100, 34).transpose(0, 2, 1)   # reshape into (951, 34, 100)
```
If you want to use the data in a "dataloader way", you can either construct your own dataset and dataloader or 
use the function `prepare_data` in the [Transformer/utils](https://github.com/aJay0422/IoT-Activity-Detection-Clean/blob/main/Transformer/utils.py), 
which returns a train dataloader and a test dataloader.

## Transformer/model.py
This file is for building the Transformer model. Three functions `transformer_base`,
`transformer_large` and `transformer_huge` return three Transformer models with
different size(different number of parameters).  
```python
from Transformer.model import transformer_base, transformer_large, transformer_huge

model_base = transformer_base()
model_large = transformer_large()
model_huge = transformer_huge()
```

## Transformer/train.py
This file is for training a Transformer model. I have trained the models of different size for 5 times
on different train test split. The model weights can be found in the folder [Transformer/model_weights](https://github.com/aJay0422/IoT-Activity-Detection-Clean/tree/main/Transformer/model_weights)
or [download](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing) here.
```python
import torch
from Transformer.model import transformer_base

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = transformer_base()
model.to(device)
weight_path = "path/to/weight"
model.load_state_dict(torch.load(weight_path, map_location=device))
```

## Transformer/weighted train.py
The idea is base on that our keypoint feature estimation may not be accurate. We can use the confidence scores 
detectron2 returns to calculate a single confidence score for each video. For those videos with high confidence,
we should give them more weights when training.  
  
For each keypoint in each frame in each video, we have one confidence score. Therefore, for each video,
we have __17 * n_frames__ confidence scores. Since the keypoint features we're using were interpolated in the temporal
dimension, we do the same to the confidence score. In the end, the number of scores in each video is 
__17 * 100__. Our goal is to turn these 1700 scores into one single score for each video, in a clever way.
  
The interpolated confidence scores can be found in the folder [feature_archive](https://github.com/aJay0422/IoT-Activity-Detection-Clean/tree/main/feature_archive)
or [download](https://drive.google.com/drive/folders/1Wmhi-ftV_buR9jFlPW0u4IR6F3idyr5G?usp=sharing) here.

```python
# How to use interpolated confidence score
import numpy as np

scores = np.load("feature_archive/confidence_scores_by_frame.npy")   # shape(951, 100, 17)
```
  
To use the weights in the training process, we need to use a new dataloader, which not only `X_batch` and `Y_batch`
at each iteration but also `weight_batch` which contains the weight for each sample within this batch. Please see
details in `mydataset_w_weight` in [Transformer/weighted train](https://github.com/aJay0422/IoT-Activity-Detection-Clean/blob/main/Transformer/weighted%20train.py).