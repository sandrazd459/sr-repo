import os 
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from IMDN import IMDN

# define dataset
class MyDataset(Dataset):
    def __init__(self, low_root, high_root):
        # image directory
        self.low_root = low_root
        self.high_root = high_root
        # image file name list
        self.low_list = os.listdir(low_root)
        self.high_list = os.listdir(high_root)
        # order sort
        self.low_list.sort()
        self.high_list.sort()
        self.len = len(self.low_list)

    def __len__(self):
        return self.len

    # get image tensor pair (low_res, high_res) with specified index 
    def __getitem__(self, index):
        low_path = os.path.join(self.low_root, self.low_list[index])
        high_path = os.path.join(self.high_root, self.high_list[index])
        # '1' indicates images are converted to black and white
        low = Image.open(low_path).convert('1')
        high = Image.open(high_path).convert('1')
        # to numpy array
        low_np = np.array(low)
        high_np = np.array(high)
        # to tensor
        low_tensor = torch.from_numpy(low_np).to(torch.float32).unsqueeze(0)
        high_tensor = torch.from_numpy(high_np).to(torch.float32).unsqueeze(0)
        return low_tensor, high_tensor

## setting up
dl = DataLoader(MyDataset('./low', './high'), batch_size = 8, shuffle = True)
# if use GPU to train, device = 'cuda', else device = 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = IMDN()
model.to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.L1Loss()

# define train function
def train(model, dl, optimizer, criterion, epochs):
    # metric for record loss and acc
    metric = {'train_loss': []}
    # train
    for epoch in tqdm(range(epochs)):
        # get the data from dataloader
        # x is low resolution image, y is high resolution image
        for i, (x, y) in enumerate(dl):
            # move data to device
            x = x.to(device)
            y = y.to(device)
            # clear gradients
            optimizer.zero_grad()
            # forward
            y_pred = model(x)
            # calculate loss
            loss = criterion(y_pred, y)
            # clear to avoid accumulation
            optimizer.zero_grad()
            # backward
            loss.backward()
            # update parameters
            optimizer.step()
            
            if i % 100 == 0 and i != 0:
                print('epoch: %d, step: %d, loss: %f'% (epoch+1, i+1, loss.item()))
                # record loss
                metric['train_loss'].append(loss.item())
        if epoch % 100 == 0 and epoch != 0:
            # save model
            torch.save(model.state_dict(), './model.pth')
    return metric

train(model, dl, optimizer, criterion, 10000)
