import numpy as np
import torch
import cv2
import pathlib
import os

from PIL import Image, ImageChops
from ai.utils import compute_image_gradients
from env import HEIGHT, WIDTH, BATCH_SIZE, IN_CHANNEL


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
        self.data_list = self._read_image_list()
        
        # retrieve the number of dataset
        self.n = len(self.data_list[0])
        for lst in self.data_list:
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
        
        # shuffle dataset
        for lst in self.data_list:
            np.random.shuffle(lst)
        
        # construct batch of images
        for b in range(self.num_batch):
            start = b * (BATCH_SIZE//len(self.categories))
            end = min((b+1) * (BATCH_SIZE//len(self.categories)), self.n)
            
            # placeholder of batch
            x_batch = np.zeros(((end - start)*len(self.categories), 8, IN_CHANNEL, HEIGHT, WIDTH))
            y_batch = np.zeros(((end - start)*len(self.categories),))
            
            # index
            i = 0
            
            for c, lst in enumerate(self.data_list):
                for d in lst[start:end]:
                    j = 0
                    
                    # load images from file
                    for f in pathlib.Path(d).glob("*.jpg"):
                        img = self._load_one_image(str(f))
                        x_batch[i, j, -3:] = img
                        j += 1
                        if j == 8:
                            break
                            
                    assert j == 8, f"j: {j}, d: {str(d)}"
                            
                    y_batch[i] = c
                    i += 1
                    
#             x_batch = self._compute_gradients(x_batch[:, :, -3:].reshape(-1, 3, HEIGHT, WIDTH)).reshape((end - start)*len(self.categories), 8, IN_CHANNEL, HEIGHT, WIDTH)
                
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
        lst = []
        
        # read image list in each category directory
        for cat in self.categories:

            # list for storing images in single category
            cat_list = []

            # read names of image in data directory
            for p in pathlib.Path(os.path.join(self.data_path, cat).replace("\\", "/")).glob("*"):
                if os.path.isdir(str(p)):
                    cat_list.append(str(p))
                
            lst.append(cat_list)
                
        return lst
    
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
        if np.random.rand() < 0.4:
            angle = np.random.randn() * 25
            image = image.rotate(angle)
        
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
