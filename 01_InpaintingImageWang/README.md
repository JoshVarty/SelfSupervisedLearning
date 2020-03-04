# Image Inpainting

**Hypothesis**: By training a network on the task of image inpainting, we are left with a set of weights that outperform randomly initialized weights on a downstream task.

**Result**: True

- Random weights baseline: **54.0%** accuracy
- Pretext weights with head fine-tuning: **56.3%** accuracy


**Methodology**:

We train a U-Net with an `xresnet34` backbone on the task of image inpainting in which it is tasked with "filling in" missing patches that have been cutout from an image.

![](https://joshvarty.files.wordpress.com/2020/02/inpainting-3.png)

We take the `xresnet34` network, add a `torch.nn.Linear()` to it and train/validate it on the ImageWang dataset.

