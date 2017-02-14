from glob import glob
import numpy as np
import scipy.misc
import os


class LSUN(object):
    def __init__(self, path, cat=None):
        self.path = path
        self.train_images = np.empty((0))
        self.train_label = None
        if cat:
            self.categories = cat
        else:
            self.categories = ["bedroom", "bridge", "church outdoor", "classroom", "conference room", "dining_room",
                               "kitchen",
                               "living room", "restaurant", "tower"]

    def _download_data(self, url, cat):
        pass

    def load_train(self):
        for c in self.categories:
            imagesPath = glob(os.path.join(self.path, c, "**", "*.webp"), recursive=True)
            for imPath in imagesPath:
                im = self.imread(imPath)
                im2=self.transform(im,im.shape[0],im.shape[1],108,108)
                im_name="/share/jproject/mbirla/lsun2/lsun/data2/"+c+"/"+imPath.split("/")[-1]
                print(im_name)
                scipy.misc.imsave(im_name,im2)
    def imread(self, path, is_grayscale=False):
        if (is_grayscale):
            return scipy.misc.imread(path, flatten=True).astype(np.float)
        else:
            return scipy.misc.imread(path).astype(np.float)

    def transform(self, image, input_height, input_width,
                  resize_height=64, resize_width=64, is_crop=True):
        if is_crop:
            cropped_image = self.center_crop(
                image, input_height, input_width,
                resize_height, resize_width)
        else:
            cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
        return np.array(cropped_image) / 127.5 - 1.

    def center_crop(self, x, crop_h, crop_w,
                    resize_h=64, resize_w=64):
        if crop_w is None:
            crop_w = crop_h
        h, w = x.shape[:2]
        j = int(round((h - crop_h) / 2.))
        i = int(round((w - crop_w) / 2.))
        return scipy.misc.imresize(
            x[j:j + crop_h, i:i + crop_w], [resize_h, resize_w])


l=LSUN("/share/jproject/mbirla/lsun2/lsun/data",["church_outdoor"])
l.load_train()