import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

from ai.AutoEncoder import AutoEncoder
from ai.DataLoader2 import DataLoader
from env import *

ETA = 1e-3
EPOCHS = 500

index = 0

print(torch.cuda.is_available())

train_dloader = DataLoader(TRAIN_DATA_PATH, TRASH_CAT, noise=False)
valid_dloader = DataLoader(VALID_DATA_PATH, TRASH_CAT, noise=False)

model = AutoEncoder(index, is_train=True)
model = nn.DataParallel(model).cuda()

optimizer = optim.Adam(model.parameters(), lr=ETA)
criterion_ce = nn.NLLLoss()
criterion_ae = nn.MSELoss()

# model.module.load(AE_CKPT_PATH)

top_valid_acc = 0.0

for e in range(EPOCHS):
    
    train_loss = 0.0
    train_clf_acc = 0.0
    
    model.train()
    
    for x_batch_, y_batch_ in train_dloader.next_batch():
        x_batch_ = x_batch_.reshape(-1, IN_CHANNEL, HEIGHT, WIDTH)
        y_batch = np.repeat(y_batch_, 8, axis=0)
        y_batch = torch.LongTensor(y_batch).cuda()
        
        x_batch_noise = np.zeros_like(x_batch_)
        r = np.random.rand(x_batch_.shape[0])
        noise = np.random.randn(*x_batch_.shape) * 0.01
        x_batch_noise[r < 0.5] = x_batch_[r < 0.5] + noise[r < 0.5]
        x_batch_noise[r >= 0.5] = x_batch_[r >= 0.5]
        
        x_batch = torch.FloatTensor(x_batch_).cuda()
        x_batch_noise = torch.FloatTensor(x_batch_noise).cuda()
        
        orig1, orig2, latent1, latent2, logps = model(x_batch_noise)
        loss_ce = criterion_ce(logps, y_batch)
        loss_ae1 = criterion_ae(latent1, orig1)
        loss_ae2 = criterion_ae(latent2, orig2)
        
        loss = loss_ce + loss_ae1 + loss_ae2
        train_loss += loss.item()
        
        with torch.no_grad():
            ps = torch.exp(logps)
            val_k, cls_k = ps.topk(1, dim=1)
            equal = cls_k == y_batch.view(*cls_k.size())
            train_clf_acc += torch.mean(equal.type(torch.FloatTensor))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (e+1)%10 == 0:
        
        valid_loss = 0.0
        valid_clf_acc = 0.0
        
        with torch.no_grad():
            model.eval()
            
            for x_batch_, y_batch_ in valid_dloader.next_batch():
                x_batch_ = x_batch_.reshape(-1, IN_CHANNEL, HEIGHT, WIDTH)
                y_batch = np.repeat(y_batch_, 8, axis=0)
                x_batch = torch.FloatTensor(x_batch_).cuda()
                y_batch = torch.LongTensor(y_batch).cuda()
        
                x_batch_noise = np.zeros_like(x_batch_)
                r = np.random.rand(x_batch_.shape[0])
                noise = np.random.randn(*x_batch_.shape) * 0.01
                x_batch_noise[r < 0.5] = x_batch_[r < 0.5] + noise[r < 0.5]
                x_batch_noise[r >= 0.5] = x_batch_[r >= 0.5]

                x_batch = torch.FloatTensor(x_batch_).cuda()
                x_batch_noise = torch.FloatTensor(x_batch_noise).cuda()

                orig1, orig2, latent1, latent2, logps = model(x_batch_noise)
                loss_ce = criterion_ce(logps, y_batch)
                loss_ae1 = criterion_ae(latent1, orig1)
                loss_ae2 = criterion_ae(latent2, orig2)
        
                loss = loss_ce + loss_ae1 + loss_ae2
                valid_loss += loss.item()
                
                ps = torch.exp(logps)
                val_k, cls_k = ps.topk(1, dim=1)
                equal = cls_k == y_batch.view(*cls_k.size())
                valid_clf_acc += torch.mean(equal.type(torch.FloatTensor))

            train_loss /= len(train_dloader)
            train_clf_acc /= len(train_dloader)
            valid_loss /= len(valid_dloader)
            valid_clf_acc /= len(valid_dloader)

            
            '''
            test_image1 = x_batch_[0, :3]
            test_image1 = test_image1*256 + 128
            test_image1 = test_image1.astype(np.ubyte)
            test_image1 = np.transpose(test_image1, axes=[1, 2, 0])

            reconstructed_image1 = reconstructed.cpu().detach().numpy()[0, :3]
            reconstructed_image1 = reconstructed_image1*256 + 128
            reconstructed_image1 = reconstructed_image1.astype(np.ubyte)
            reconstructed_image1 = np.transpose(reconstructed_image1, axes=[1, 2, 0])

            test_image2 = x_batch_[0, 3:]
            test_image2 = test_image2*256 + 128
            test_image2 = test_image2.astype(np.ubyte)
            test_image2 = np.transpose(test_image2, axes=[1, 2, 0])

            reconstructed_image2 = reconstructed.cpu().detach().numpy()[0, 3:]
            reconstructed_image2 = reconstructed_image2*256 + 128
            reconstructed_image2 = reconstructed_image2.astype(np.ubyte)
            reconstructed_image2 = np.transpose(reconstructed_image2, axes=[1, 2, 0])

            print("Original1:")
            plt.imshow(test_image1)
            plt.show()

            print("Reconstructed1:")
            plt.imshow(reconstructed_image1)
            plt.show()

            print("Original2:")
            plt.imshow(test_image2)
            plt.show()

            print("Reconstructed2:")
            plt.imshow(reconstructed_image2)
            plt.show()
            '''

            print(f"Epochs: {e+1}/{EPOCHS}")
            print(f"Train loss: {train_loss:.8f}")
            print(f"Train acc: {train_clf_acc:.8f}")
            print(f"Valid loss: {valid_loss:.8f}")
            print(f"Valid acc: {valid_clf_acc:.8f}")
            
            if top_valid_acc < valid_clf_acc:
                top_valid_acc = valid_clf_acc
                model.module.save(AE_CKPT_PATH)

# model.save()