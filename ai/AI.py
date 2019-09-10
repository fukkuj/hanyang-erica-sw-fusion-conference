import torch
import numpy as np

from ai.TrashClassifierAE import TrashClassifier
from env import *


class AI():

    def __init__(self):

        self.classifier = None
        self.detector = None

        self.classifier_model = None
        self.detector_model = None

    def build(self):

        print("Building AI module...")
        self.classifier = TrashClassifier(fine_tune=False)
        self.detector = TrashDetector()

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.classifier = torch.nn.DataParallel(self.classifier, device_ids=CUDA_DEVICES)
                self.detector = torch.nn.DataParallel(self.detector, device_ids=CUDA_DEVICES)

                self.classifier_model = self.classifier.module
                self.detector_model = self.detector.module
            else:
                self.classifier_model = self.classifier
                self.detector_model = self.detector
        print("AI module was built.")

    def predict(self, x):
        """
        Arguments:
        ----------
        :x images, shaped of (1, 8, 6, HEIGHT, WIDTH)
        """

        with torch.no_grad():
            x = torch.FloatTensor(x)
            
            if detect(x):
                result = int(classify(x))
            else:
                result = -1
                
        return result
            
    def detect(self, x):
        x = x.view(-1, IN_CHANNEL, HEIGHT, WIDTH)
            
        logps = self.detector(x)
        ps = torch.exp(logps)
        
        ratio = ps[:, 1] >= 0.95
        ratio = torch.mean(ratio.type(torch.FloatTensor))
        
        # if 6 or more pictures have trash,
        if ratio > 0.7:
            return True
        else:
            return False
        
    def classify(self, x):
        logps = self.classifier(x)
        ps = torch.exp(logps)
        
        _, topk = ps.topk(1, dim=1)
        return topk.cpu().detach().numpy().squeeze()[0]
