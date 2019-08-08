#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from std_msgs.msg import UInt8MultiArray

HEIGHT = 128
WIDTH = 128
CHANNEL = 3
NUM_STEP = 8


class ImageSubscriber():
    """
    Retrieve image from UsbCamera
    
    UsbCamera로부터 이미지를 수신
    """

    def __init__(self):
        self.sub = rospy.Subscriber("image_data", UInt8MultiArray, self.image_callback, queue_size=None)
        self.image_data = []
        self.ready = True
        self.cnt = 0
        
        self.two = False

    def get_image(self):
        if self.two is False:
            return None
        
        if len(self.image_data) == 0:
            return None

        img1 = self.image_data.pop(0)
        img2 = self.image_data.pop(0)
        
        return img1, img2

    def image_callback(self, data):

        if self.ready is False:
            return

        rospy.loginfo("image received.")
        
        image = np.zeros((HEIGHT * WIDTH * CHANNEL,))

        for i, c in enumerate(data.data):
            image[i] = ord(c)

        self.image_data.append(image.reshape(HEIGHT, WIDTH, CHANNEL))
        self.cnt += 1

        if self.cnt == 8:
            self.ready = False
            
        if len(image_data) >= 2:
            self.two = True
        else:
            self.two = False
