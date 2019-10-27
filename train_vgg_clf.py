import torch
import numpy as np

from torch import nn, optim
from ai.ClassifierVGG import ClassifierVGG
from env import *
from ai.DataLoader import DataLoader


ETA = 1e-4
EPOCHS = 10

def main():
    train_loader = DataLoader(TRASH_TRAIN_DATA_PATH, TRASH_CAT, noise=True)
    valid_loader = DataLoader(TRASH_VALID_DATA_PATH, TRASH_CAT, noise=True)

    model = ClassifierVGG(fine_tune=False).cuda()
    model.load(VGG_CLF_CKPT_PATH)

    optimizer = optim.Adam(model.parameters(), lr=ETA)
    criterion = nn.NLLLoss()

    top_valid_acc = 0.0

    for e in range(EPOCHS):

        train_loss = 0.0
        train_acc = 0.0
        
        for x, y in train_loader.next_batch():
            x = torch.FloatTensor(x).cuda()
            y = torch.LongTensor(y).cuda()

            logps = model(x)
            loss = criterion(logps, y)
            train_loss += loss.item()

            with torch.no_grad():
                ps = torch.exp(logps)
                val_k, top_k = ps.topk(1, dim=-1)
                equality = top_k == y.view(*top_k.size())
                train_acc += torch.mean(equality.type(torch.FloatTensor))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        with torch.no_grad():
            valid_loss = 0.0
            valid_acc = 0.0

            model.eval()

            for x, y in valid_loader.next_batch():
                x = torch.FloatTensor(x).cuda()
                y = torch.LongTensor(y).cuda()

                logps = model(x)
                loss = criterion(logps, y)
                valid_loss += loss.item()

                ps = torch.exp(logps)
                val_k, top_k = ps.topk(1, dim=-1)
                equality = top_k == y.view(*top_k.size())
                valid_acc += torch.mean(equality.type(torch.FloatTensor))

            valid_loss /= len(valid_loader)
            valid_acc /= len(valid_loader)

            model.train()

        print(f"Epochs {e+1}/{EPOCHS}")
        print(f"Train loss: {train_loss:.8f}")
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Valid loss: {valid_loss:.8f}")
        print(f"Valid accuracy: {valid_acc:.4f}")

        if top_valid_acc < valid_acc:
            top_valid_acc = valid_acc
            model.save(VGG_CLF_CKPT_PATH)

if __name__ == "__main__":
    main()
