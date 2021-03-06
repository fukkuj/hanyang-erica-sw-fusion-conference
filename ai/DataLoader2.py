import numpy as np
import torch
import cv2
import pathlib
import os

from PIL import Image, ImageChops
from env import *


class DataLoader():
    """
    Preprocessor of images.
    Load images from data path.
    """
    
    def __init__(self, data_path, categories, noise=True):
        self.data_path = data_path
        self.categories = categories
        self.noise = noise

        # build DataLoader. read all images list from data directory.
        # but, does not load actual image data. just load list of names of image.
        self.data_list1, self.data_list2 = self._read_image_list()
        
        # retrieve the number of dataset
        self.n = len(self.data_list1[0])
        for lst in self.data_list1:
            self.n = min(self.n, len(lst))
        
        # compute the number of batch
        self.num_batch = self.n // (8*(BATCH_SIZE//len(self.categories)))
        
        # print information of DataLoader instance.
        print(f"Number of data: {self.n}")
        print(f"Number of batch: {self.num_batch}")
        
    def __len__(self):
        """
        Returns number of batch.
        """

        return self.num_batch
    
    def next_batch(self):
        """
        data generator.
        if this method is called, load batch_size images, preprocess them, and yield them.

        Returns:
        --------
        :x_batch image data, shaped of (batch_size, channel, height, width)
        :y_batch label for data, shaped of (batch_size,)
        """
        
        data_list1 = []
        data_list2 = []
        
        # shuffle dataset
        for lst1, lst2 in zip(self.data_list1, self.data_list2):
            assert len(lst1) == len(lst2)
            r = np.arange(len(lst1)).astype(np.int32)
            np.random.shuffle(r)
            
            cat_list1 = []
            cat_list2 = []
            
            for i in r.tolist():
                cat_list1.append(lst1[i])
                cat_list2.append(lst2[i])
                
            data_list1.append(cat_list1)
            data_list2.append(cat_list2)
        
        # construct batch of images
        for b in range(self.num_batch):
            start = b * (BATCH_SIZE*8//len(self.categories))
            end = (b+1) * (BATCH_SIZE*8//len(self.categories))
            
            # placeholder of batch
            x_batch = np.zeros(((end - start)*len(self.categories)//8, 8, IN_CHANNEL, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start)*len(self.categories)//8,))
            
            # index
            i = 0
            
            for c in range(len(self.categories)):
                lst1 = data_list1[c]
                lst2 = data_list2[c]

                j = 0
                
                for img_path1, img_path2 in zip(lst1[start:end], lst2[start:end]):
                    img1 = self._load_one_image(img_path1)
                    img2 = self._load_one_image(img_path2)

                    if np.random.rand() > 0.5:
                        x_batch[i, j, :3] = img1
                        x_batch[i, j, 3:] = img2
                    else:
                        x_batch[i, j, :3] = img2
                        x_batch[i, j, 3:] = img1

                    j += 1
                    if j == 8:
                        y_batch[i] = c
                        i += 1
                        j = 0
                
            # generate data
            yield x_batch, y_batch
    
    def _read_image_list(self):
        """
        Read image list from data folder.
        But, not load actual images, just read list of images.

        Returns:
        --------
        :lst list of all images in data folder
        """
        
        # list of all images in data folder
        lst1 = []
        lst2 = []
        
        # read image list in each category directory
        for cat in self.categories:

            # list for storing images in single category
            cat_list1 = []
            cat_list2 = []

            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat, "1").replace("\\", "/")).glob("*.jpg"):
                cat_list1.append(str(p))
                
            cat_list1 = sorted(cat_list1)
            lst1.append(cat_list1)
                    
            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat, "2").replace("\\", "/")).glob("*.jpg"):
                cat_list2.append(str(p))
                
            cat_list2 = sorted(cat_list2)
            lst2.append(cat_list2)
                
        return lst1, lst2
    
    def _load_one_image(self, path):
        """
        Given path of a single image, load that actual image into numpy array.

        Arguments:
        ----------
        :path path of a single image

        Returns:
        --------
        :image numpy array storing image data
        """

        # open image
        image = Image.open(path)
        
        # resize
        image = image.resize((WIDTH, HEIGHT))
        
        # random translation
        #if np.random.rand() < 0.4:
        #    x_offset = np.random.randint(0, 10)
        #    y_offset = np.random.randint(0, 10)
        #    image = ImageChops.offset(image, x_offset, y_offset)
            
        if np.random.rand() < 0.5:
            image.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.rand() < 0.5:
            image.transpose(Image.FLIP_TOP_BOTTOM)
        if np.random.rand() < 0.2:
            image.transpose(Image.ROTATE_90)
        if np.random.rand() < 0.2:
            image.transpose(Image.ROTATE_180)
        if np.random.rand() < 0.2:
            image.transpose(Image.ROTATE_270)
            
        # random rotation
        # if np.random.rand() < 0.4:
        #     angle = np.random.randn() * 25
        #     image = image.rotate(angle)
        
        # into numpy array
        image = np.array(image).astype(np.float32)
        
        # into (1, channel, height, width)
        image = np.transpose(image, axes=(2, 0, 1))
        image = image.reshape(1, *image.shape)
        
        #########
        # add noise
        if self.noise is True:
            if np.random.rand() < 0.6:
                noise = np.random.randn(*image.shape) * 10
                image = image + noise
            
        image[image > 255] = 255
        image[image < 0] = 0
            
        # standardize
        image = (image - 128) / 256
        #########
        
        return image

