# UNOSAT Challenge 2nd place solution

## Installation
`pip install -r requirements.txt`

## Files
* **conf.py** - paths to all data/model files, set proper paths before executing `train.py` and `inference.py`
* **constants.py** - constant values used for for inference and training
* **data_conf.py** - basic module for loading imagery and masks, with normalization
* **data_provider.py** - efficient data providers for training/testing
* **inference.py** - module for predictions inference
* **metrics.py** - score and loss functions used for training/evaluation
* **models.py** - neural zoo - different semantic segmentation models
* **mv_extr.py** - model for extraction of basic statistics of data
* **train.py** - training script


## Workflow

### Data
I tried many different normalization/cleaning approaches. Denoising, smoothing, averaging, clipping to percentiles. All in all most efficient method was clipping data to `[0, 2]` range and then normalizing to empirically computed variance and mean for each layer. Only 2 channels were used - `vv` and `vh`. I didn't create any other input layers. 
For training I was heavily augmenting data - rotations, flipping, transposing, shifting, scaling, cropping, grid distorting.

## Cross validation
In each fold 4 images of 3 cities were used as training data, and 4th city was used as evaluation. Test score was average of 4 folds scores and 4 models were obtained.

## Model 
I tried many different models. Best model was UNet with pretrained VGG11 encoder. First layer was VGG11 convolutional kernel reduced to 2 channels. (First layer of VGG encdoder has kernel for RGB imagery so I had to reduce its shape from `(64, 3, 3, 3)` to `64, 2, 3, 3`)

## Loss function
Weighted average of `logloss` and `dice loss`.

## Optimizer
Adam with initial `lr=3e-4`
https://twitter.com/karpathy/status/801621764144971776 :)

## LR Scheduler
Learning Rate with exponential reduction on plateau, with `0.25` factor and `4` patience.

## Inference
For each city I predicted masks for each image separately on all 4 models obtained during `4-fold` training.
Final prediction was blend of `16` logit predictions (4 images * 4 models).

Prediction was done with sliding window approach - **i think it gave biggest boost for my score**.
I was doing prediction over a big region but was saving only interior of each, because predictions were not that accurate on the border.
https://i0.wp.com/blog.kaggle.com/wp-content/uploads/2017/04/image16.png?w=460

## What didn't work
* **UNet with ResNet encoder** - I expected bigger filter and stride in first layer will be beneficial for imagery of that size - it wasn't.
* **LinkNet** - I expected that lightweight network which can process bigger images will be better
* More lightweight architectures didn't work, I didn't tried bigger architectures that much
* `vv/vh` layer
* Smarter ways of preprocessing data - clipping to percentiles, averaging layer before pushing to network
* *Neural compression* - new brilliant idea for satellite imagery - https://www.kaggle.com/c/understanding_cloud_organization/discussion/118255#latest-678189
## What's worth trying
* **Cosine annealing with warm starts** - I had impression that reducing LR on plateu wasn't that efficient
* **OOF ensembling** - finding better `0/1` threshold would be beneficial for competition with f1 score
