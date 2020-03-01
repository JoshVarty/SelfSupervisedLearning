import numpy as np
from fastai2.vision.all import PILImage, Image
from fastai2.vision.augment import RandTransform


# We create this dummy class in order to create a transform that ONLY operates on images of this type
# We will use it to create all input images
class PILImageInput(PILImage):
    pass


class RandomCutout(RandTransform):
    "Picks a random scaled crop of an image and resize it to `size`"
    split_idx = None

    def __init__(self, min_n_holes=5, max_n_holes=10, min_length=5, max_length=50, **kwargs):
        super().__init__(**kwargs)
        self.min_n_holes = min_n_holes
        self.max_n_holes = max_n_holes
        self.min_length = min_length
        self.max_length = max_length

    def encodes(self, x: PILImageInput):
        """
        Note that we're accepting our dummy PILImageInput class
        fastai2 will only pass images of this type to our encoder.
        This means that our transform will only be applied to input images and won't
        be run against output images.
        """

        n_holes = np.random.randint(self.min_n_holes, self.max_n_holes)
        pixels = np.array(x)  # Convert to mutable numpy array. FeelsBadMan
        h, w = pixels.shape[:2]

        for n in range(n_holes):
            h_length = np.random.randint(self.min_length, self.max_length)
            w_length = np.random.randint(self.min_length, self.max_length)
            h_y = np.random.randint(0, h)
            h_x = np.random.randint(0, w)
            y1 = int(np.clip(h_y - h_length / 2, 0, h))
            y2 = int(np.clip(h_y + h_length / 2, 0, h))
            x1 = int(np.clip(h_x - w_length / 2, 0, w))
            x2 = int(np.clip(h_x + w_length / 2, 0, w))

            pixels[y1:y2, x1:x2, :] = 0

        return Image.fromarray(pixels, mode='RGB')