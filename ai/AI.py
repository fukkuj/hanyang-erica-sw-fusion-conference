import torch
import numpy as np

from ai import TrashClassifier, TrashDetector
from env import CUDA_DEVICES


class AI():
    
    def __init__(self):
        
        self.classifier = None
        self.detector = None
        
        self.classifier_model = None
        self.detector_model = None
        
    def build(self):
        
        print("Building AI module...")
        self.classifier = TrashClassifier()
        self.detector = TrashDetector()
        
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                self.classifier = torch.nn.DataParallel(self.classifier, device_ids=CUDA_DEVICES)
                self.detector = torch.nn.DataParallel(self.detector, device_ids=CUDA_DEVICES)
                
                self.classifier_model = self.classifier.module
                self.detector_model = self.detector.modules()
            else:
                
        print("AI module was built.")
        
    def inference(self, x):
        """
        Arguments:
        ----------
        :x images, shaped of (8, 6, HEIGHT, WIDTH)
        """
        
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.float)
            if torch.cuda.is_available():
                pass
            
            logps = self.detector(x)
            ps = torch.exp(logps)
