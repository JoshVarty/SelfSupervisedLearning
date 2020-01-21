import os
import json

from fastai2.basics import DataBlock, RandomSplitter

from fastai2.vision.all import ImageBlock, PILMask, get_image_files
from fastai2.vision.core import get_annotations
from fastai2.vision.augment import aug_transforms, RandomResizedCrop
from fastai2.vision.learner import unet_learner, unet_config

from fastai2.data.external import download_url, URLs, untar_data

from torch.nn import MSELoss
from torchvision.models import resnet34

pascal_path = untar_data(URLs.PASCAL_2012)

data = DataBlock(blocks=(ImageBlock, ImageBlock),
                 get_items=get_image_files,
                 splitter=RandomSplitter(),
                 get_y=lambda o: o)

databunch = data.databunch(pascal_path/'train',
                           bs=8,
                           item_tfms=RandomResizedCrop(460, min_scale=0.75),
                           batch_tfms=[*aug_transforms(size=299, max_warp=0)])

# HACK: We're predicting pixel values, so we're just going to predict a single output class
databunch.vocab = ['R', 'G', 'B']

loss = MSELoss()
learn = unet_learner(databunch, resnet34, config=unet_config(), loss_func=loss)

learn.fit_one_cycle(1, 1e-3)

x = 6