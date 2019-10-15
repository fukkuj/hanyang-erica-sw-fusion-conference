import numpy as np
import torch
import cv2
import pathlib
import os

from PIL import Image, ImageChops
from ai.utils import compute_image_gradients
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
        self.num_batch = int(np.ceil(self.n / (BATCH_SIZE//len(self.categories))))
        
        # print information of DataLoader instance.
        print(f"Number of data batch: {self.n}")
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
            start = b * (BATCH_SIZE//len(self.categories))
            end = min((b+1) * (BATCH_SIZE//len(self.categories)), self.n)
            
            # placeholder of batch
            x_batch = np.zeros(((end - start)*len(self.categories), 8, IN_CHANNEL, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start)*len(self.categories),))
            
            # index
            i = 0
            
            for c in range(len(self.categories)):
                lst1 = data_list1[c]
                lst2 = data_list2[c]
                
                for d1, d2 in zip(lst1[start:end], lst2[start:end]):
                    j = 0
                    r = np.random.rand()
                    
                    # load images from file
                    for f in pathlib.Path(d1).glob("*.jpg"):
                        img = self._load_one_image(str(f))
                        if r < 0.5:
                            x_batch[i, j, :3] = img
                        else:
                            x_batch[i, j, 3:] = img
                        j += 1
                        if j == 8:
                            break
                            
                    assert j == 8, f"j: {j}, d: {str(d1)}"
                            
                    j = 0
                    
                    # load images from file
                    for f in pathlib.Path(d2).glob("*.jpg"):
                        img = self._load_one_image(str(f))
                        if r < 0.5:
                            x_batch[i, j, 3:] = img
                        else:
                            x_batch[i, j, :3] = img
                        j += 1
                        if j == 8:
                            break
                            
                    assert j == 8, f"j: {j}, d: {str(d1)}"
                            
                    y_batch[i] = c
                    i += 1
                
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
            for p in pathlib.Path(os.path.join(self.data_path, cat, "collected_1").replace("\\", "/")).glob("*"):
                if os.path.isdir(str(p)):
                    cat_list1.append(str(p))
                
            lst1.append(cat_list1)
                    
            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat, "collected_2").replace("\\", "/")).glob("*"):
                if os.path.isdir(str(p)):
                    cat_list2.append(str(p))
                
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
        if np.random.rand() < 0.4:
            x_offset = np.random.randint(0, 10)
            y_offset = np.random.randint(0, 10)
            image = ImageChops.offset(image, x_offset, y_offset)
            
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
    
    def _compute_gradients(self, images):
        """
        compute gradients of an image to use gradients by image features.

        Arguments:
        ----------
        :images images data, numpy array

        Returns:
        --------
        :res new image array, this array has new information about gradients of images.
        """
        n = images.shape[0]
        
        # result
        res = np.zeros((n, 9, HEIGHT, WIDTH))
        
        # compute gradients
        res[:, :6] = compute_image_gradients((images.astype(np.float32) - 128) / 128)
        
        # add noise
        if self.noise is True:
            r = np.random.rand(n)
            noise = np.random.randn(*images.shape) * 10
            noise[r < 0.4] = 0
            images = images + noise
            
        # standardize
        images = (images.astype(np.float32) - 128) / 128
        res[:, 6:] = images
        
        return res


class DetectorDataLoader():
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
        for i in range(len(self.data_list1)):
            self.n = min(self.n, len(self.data_list1[i]))
        
        # compute the number of batch
        self.num_batch = int(np.ceil(self.n / (BATCH_SIZE//len(self.categories))))
        
        # print information of DataLoader instance.
        print(f"Number of data batch: {self.n}")
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
            start = b * (BATCH_SIZE//len(self.categories))
            end = min((b+1) * (BATCH_SIZE//len(self.categories)), self.n)
            
            # placeholder of batch
            x_batch = np.zeros(((end - start)*len(self.categories), IN_CHANNEL, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start)*len(self.categories),))
            
            # index
            i = 0
            
            for c in range(len(self.categories)):
                lst1 = data_list1[c]
                lst2 = data_list2[c]
                
                for f1, f2 in zip(lst1[start:end], lst2[start:end]):
                    img1 = self._load_one_image(f1)
                    img2 = self._load_one_image(f2)

                    x_batch[i, :3] = img1
                    x_batch[i, 3:] = img2
                            
                    y_batch[i] = c
                    i += 1
                
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
            
            cat_list1 = []
            cat_list2 = []

            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat, "1").replace("\\", "/")).glob("*.jpg"):
                cat_list1.append(str(p))
                    
            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat, "2").replace("\\", "/")).glob("*.jpg"):
                cat_list2.append(str(p))

            lst1.append(cat_list1)
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
        if np.random.rand() < 0.4:
            x_offset = np.random.randint(0, 10)
            y_offset = np.random.randint(0, 10)
            image = ImageChops.offset(image, x_offset, y_offset)
            
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
            
        # standardize
        image = (image - 128) / 128
        #########
        
        return image
    
    def _compute_gradients(self, images):
        """
        compute gradients of an image to use gradients by image features.

        Arguments:
        ----------
        :images images data, numpy array

        Returns:
        --------
        :res new image array, this array has new information about gradients of images.
        """
        n = images.shape[0]
        
        # result
        res = np.zeros((n, 9, HEIGHT, WIDTH))
        
        # compute gradients
        res[:, :6] = compute_image_gradients((images.astype(np.float32) - 128) / 128)
        
        # add noise
        if self.noise is True:
            r = np.random.rand(n)
            noise = np.random.randn(*images.shape) * 10
            noise[r < 0.4] = 0
            images = images + noise
            
        # standardize
        images = (images.astype(np.float32) - 128) / 128
        res[:, 6:] = images
        
        return res
