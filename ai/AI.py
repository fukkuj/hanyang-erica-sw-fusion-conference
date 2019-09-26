import torch
import numpy as np

from ai.TrashClassifierAE import TrashClassifier
from ai.TrashDetector import TrashDetector
from env import *


class AI():

    def __init__(self):

        self.classifier = None
        self.detector = None

        self.classifier_model = None
        self.detector_model = None

    def build(self):

        print("Building AI module...")
        self.classifier = TrashClassifier(fine_tune=False).cuda()
        self.classifier.load(CLF_CKPT_PATH)
        self.detector = TrashDetector().cuda()
        self.detector.load(DET_CKPT_PATH)

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.classifier = torch.nn.DataParallel(self.classifier, device_ids=CUDA_DEVICES).cuda()
                self.detector = torch.nn.DataParallel(self.detector, device_ids=CUDA_DEVICES).cuda()

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
            x = (x - 128) / 256
            x = torch.FloatTensor(x).cuda()
            
            result = int(self.classify(x))

            #if self.detect(x):
            #    result = int(self.classify(x))
            #else:
            #    result = -1
                
        return result
            
    def detect(self, x):
        x = x.view(-1, IN_CHANNEL, HEIGHT, WIDTH)
            
        logps = self.detector(x)
        ps = torch.exp(logps)

        print(ps[:, 1])
        
        ratio = ps[:, 1] < 0.1
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
        return topk.cpu().detach().numpy().squeeze()
