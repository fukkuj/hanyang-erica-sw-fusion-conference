import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from ai.TrashClassifier import TrashClassifier
from ai.DataLoader import DataLoader
from env import *

EPOCHS = 100
ETA = 1e-4

model = TrashClassifier(fine_tune=True)
model = model.cuda()
model.load(CNN_CLF_CKPT_PATH)

train_dloader = DataLoader(TRASH_TRAIN_DATA_PATH, TRASH_CAT)
valid_dloader = DataLoader(TRASH_VALID_DATA_PATH, TRASH_CAT)

optimizer = optim.Adam(model.parameters(), lr=ETA)
criterion = nn.NLLLoss()

top_valid_acc = 0.0

for e in range(EPOCHS):
    
    train_loss = 0.0
    train_acc = 0.0
    valid_loss = 0.0
    valid_acc = 0.0
    
    for x_batch, y_batch in train_dloader.next_batch():
        x_batch = torch.FloatTensor(x_batch).cuda()
        y_batch = torch.LongTensor(y_batch).cuda()
        
        logps = model(x_batch)
        loss = criterion(logps, y_batch)
        
        with torch.no_grad():
            train_loss += loss.item()
            ps = torch.exp(logps)
            
            ps_k, cls_k = ps.topk(1, dim=1)
            equal = cls_k == y_batch.view(*cls_k.size())
            train_acc += torch.mean(equal.type(torch.FloatTensor))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (e+1) % 10 == 0:
        with torch.no_grad():
            model.eval()

            for x_batch, y_batch in valid_dloader.next_batch():
                x_batch = torch.FloatTensor(x_batch).cuda()
                y_batch = torch.LongTensor(y_batch).cuda()

                logps = model(x_batch)
                loss = criterion(logps, y_batch)

                valid_loss += loss.item()

                ps = torch.exp(logps)
                ps_k, cls_k = ps.topk(1, dim=1)
                equal = cls_k == y_batch.view(*cls_k.size())
                valid_acc += torch.mean(equal.type(torch.FloatTensor))

            train_loss /= len(train_dloader)
            train_acc /= len(train_dloader)
            valid_loss /= len(valid_dloader)
            valid_acc /= len(valid_dloader)

            print(f"Epochs: {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_acc:.8f}")
            print(f"Valid loss: {valid_loss:.8f}")
            print(f"Valid acc: {valid_acc:.8f}")
            
            if top_valid_acc < valid_acc:
                top_valid_acc = valid_acc
                model.save(CNN_CLF_CKPT_PATH)

            model.train()