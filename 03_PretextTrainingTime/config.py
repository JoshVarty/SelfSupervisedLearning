from fastai2.layers import Mish, MaxPool
from fastai2.vision.models.xresnet import xresnet34

config = {
    'lr': 8e-3,
    'size': 128,
    'sqrmom': 0.99,
    'mom': 0.9,
    'eps': 1e-6,
    'epochs': 15,
    'bs': 64,
    'opt': 'ranger',
    'sh': 0.,
    'sa': 0,
    'sym': 0,
    'beta': 0.,
    'act_fn': Mish,
    'fp16': 0,
    'pool': MaxPool,
    'runs': 1,
    'model': xresnet34
}
