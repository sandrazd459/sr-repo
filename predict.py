import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from IMDN import IMDN

# loss function
criterion = nn.L1Loss()

# define predict function
def predict(filename, model):
    # input : filename and model
    low_filename = './test_low/' + filename
    high_filename = './test_high/' + filename
    img_high = Image.open(high_filename).convert('1')
    img_high_np = np.array(img_high)
    img = Image.open(low_filename).convert('1')
    #img_cubic = img.copy() # bicubic
    img_np = np.array(img) # convert low resolution image to numpy array

    img_tensor = torch.from_numpy(img_np).to(torch.float32).unsqueeze(0).unsqueeze(0) # convert low resolution image to tensor
    img_high_tensor = torch.from_numpy(img_high_np).to(torch.float32).unsqueeze(0).unsqueeze(0) # convert low resolution image to tensor
    y_pred = model(img_tensor) # predict high resolution image

    mse = criterion(y_pred, img_high_tensor) # calculate loss
    #acc = calacc(y_pred, img_high_tensor) # calculate accuracy

    y_pred = y_pred.detach().numpy() # convert tensor to numpy array
    y_pred = np.where(y_pred > 1, 1, y_pred) # convert numpy array to binary array
    y_pred = np.where(y_pred < 0, 0, y_pred)
    y_pred = y_pred.squeeze() # remove the first dimension
    
    # plot low resolution image and predicted image and high resolution image
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(y_pred, cmap='gray')
    plt.title('L1 Loss: %.4f' % (mse), fontsize=10)
    plt.suptitle('low resolution image, predicted image, high resolution image', y=1.05, fontsize=8)
    plt.subplot(1, 3, 3)
    plt.imshow(img_high_np, cmap='gray')
    plt.show()

if __name__ == '__main__':
    model = IMDN() # define model
    model.load_state_dict(torch.load('./model.pth', map_location='cpu')) # load model checkpoint
    predict('test2.png', model)
