import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from ai.TrashDetector import TrashDetector
from ai.DataLoader2 import DataLoader
from env import *

ETA = 1e-3
EPOCHS = 300

train_dloader = DataLoader(DETECTOR_TRAIN_DATA_PATH, DETECTOR_CAT, noise=True)
valid_dloader = DataLoader(DETECTOR_VALID_DATA_PATH, DETECTOR_CAT, noise=True)

model = TrashDetector()
model = nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=ETA)
criterion = nn.NLLLoss()

top_valid_acc = 0.0

for e in range(EPOCHS):
    
    train_loss = 0.0
    train_clf_acc = 0.0
    
    for x_batch_, y_batch_ in train_dloader.next_batch():
        x_batch_ = x_batch_.reshape(-1, IN_CHANNEL, HEIGHT, WIDTH)
        y_batch = np.repeat(y_batch_, 8, axis=0)
        
        x_batch = torch.FloatTensor(x_batch_).cuda()
        y_batch = torch.LongTensor(y_batch).cuda()
        
        logps = model(x_batch)
        loss = criterion(logps, y_batch)
        train_loss += loss.item()
        
        with torch.no_grad():
            ps = torch.exp(logps)
            val_k, cls_k = ps.topk(1, dim=1)
            equal = cls_k == y_batch.view(*cls_k.size())
            train_clf_acc += torch.mean(equal.type(torch.FloatTensor))
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (e+1) % 10 == 0:
        
        valid_loss = 0.0
        valid_clf_acc = 0.0
        
        with torch.no_grad():
            model.eval()
            
            for x_batch_, y_batch_ in valid_dloader.next_batch():
                x_batch_ = x_batch_.reshape(-1, IN_CHANNEL, HEIGHT, WIDTH)
                y_batch = np.repeat(y_batch_, 8, axis=0)
                
                x_batch = torch.FloatTensor(x_batch_).cuda()
                y_batch = torch.LongTensor(y_batch).cuda()

                logps = model(x_batch)
                ps = torch.exp(logps)
                _, topk = ps.topk(1, dim=1)
                equal = topk == y_batch.view(*topk.size())
                valid_clf_acc += torch.mean(equal.type(torch.FloatTensor))
        
                loss = criterion(logps, y_batch)
                valid_loss += loss.item()

            train_loss /= len(train_dloader)
            train_clf_acc /= len(train_dloader)
            valid_loss /= len(valid_dloader)
            valid_clf_acc /= len(valid_dloader)

            print(f"Epochs: {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_clf_acc:.8f}")
            print(f"Valid loss: {valid_loss:.8f}")
            print(f"Valid acc: {valid_clf_acc:.8f}")
            
            if top_valid_acc < valid_clf_acc:
                top_valid_acc = valid_clf_acc
                model.module.save(DET_CKPT_PATH)

            model.train()
        

# print(torch.cuda.is_available())
