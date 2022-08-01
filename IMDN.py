import torch
import torch.nn as nn

ALPHA = 0.05  # negtaive slope for Leaky Relu

# Figure 3
class CCA(nn.Module):
    def __init__(self, in_channels):
        super(CCA, self).__init__()
        # contrast  = summation of sd & mean
        self.Stdv = stdv_channels
        self.Mean = nn.AdaptiveAvgPool2d(1)
        self.Conv1 = nn.Conv2d(in_channels, 4, 1, 1, 0)
        self.Conv2 = nn.Conv2d(4, 64, 1, 1, 0)
        self.Relu = nn.LeakyReLU(ALPHA)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.Stdv(x) + self.Mean(x)
        out = self.Relu(self.Conv1(out))
        out = self.Sigmoid(self.Conv2(out))
        return x * out

# helper funtions
def mean_channels(F):
    assert(F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))

def stdv_channels(F):
    assert(F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

# Figure 2
class IMDB(nn.Module):
    def __init__(self, in_channels):
        super(IMDB, self).__init__()
        self.in_channels = in_channels       # input (64)
        self.left_channels = 16              # extract due to distillation (16)
        self.right_channels = 48             # distillation leftover (48)
        # in, out, kernal size = 3, stride = 1, padding = 0 or 1
        self.Conv1 = nn.Conv2d(self.in_channels, self.in_channels, 3, 1, 1)     # (64, 64) Table 1
        self.Conv2 = nn.Conv2d(self.right_channels, self.in_channels, 3, 1, 1)  # (48, 64)
        self.Conv3 = nn.Conv2d(self.right_channels, self.in_channels, 3, 1, 1)  # (48, 64)
        self.Conv4 = nn.Conv2d(self.right_channels, self.left_channels, 3, 1, 1)# (48, 16)
        # last 1*1 convolution
        self.Conv5 = nn.Conv2d(self.in_channels, self.in_channels, 1, 1, 0)
        # leaky ReLu activation after conv1 - 4
        self.Relu = nn.LeakyReLU(ALPHA) 
        # Contrast-aware chennel attention module (CCA) Layer
        self.Cca = CCA(self.left_channels * 4) # (16 * 4)
    
    def forward(self, x):
        # progressive refinement module (PRM) 
        out_c1 = self.Relu(self.Conv1(x))
        c1_left, c1_right = torch.split(out_c1, (self.left_channels, self.right_channels), 1)
        out_c2 = self.Relu(self.Conv2(c1_right))
        c2_left, c2_right = torch.split(out_c2, (self.left_channels, self.right_channels), 1)
        out_c3 = self.Relu(self.Conv3(c2_right))
        c3_left, c3_right = torch.split(out_c3, (self.left_channels, self.right_channels), 1)
        out_c4 = self.Relu(self.Conv4(c3_right))
        out = torch.cat([c1_left, c2_left, c3_left, out_c4], 1)

        # CCA, conv, and addition
        out = self.Cca(out)
        out = x + self.Conv5(out)
        return out


# Figure 1
class IMDN(nn.Module):
    def __init__(self):
        super(IMDN, self).__init__()
        self.in_channels = 1 #?
        self.out_channels = 64

        # first conv
        #self.Conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, 1, 1) # 3,....
        self.Conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        # IMDBs
        self.IMDB1 = IMDB(self.out_channels)
        self.IMDB2 = IMDB(self.out_channels)
        self.IMDB3 = IMDB(self.out_channels)
        self.IMDB4 = IMDB(self.out_channels)
        # second & third conv
        self.Conv2 = nn.Conv2d(self.out_channels * 4, self.out_channels, 1, 1, 0)
        self.Relu = nn.LeakyReLU(ALPHA)
        self.Conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)
        self.Upsampler = nn.Sequential(
            nn.Conv2d(self.out_channels, self.in_channels * (2 ** 2), 3, 1, 1), ## enlarge by 2
            nn.PixelShuffle(2) # 
        )

        #self.Bicubic = nn.functional.interpolate(1, scale_factor=2, mode='bilinear')
        self.Bilinear = nn.UpsamplingBilinear2d(scale_factor=2)
    
    def forward(self, x):
        # first conv
        out_c1 = self.Conv1(x)
        # IMDBs
        out_b1 = self.IMDB1(out_c1)
        out_b2 = self.IMDB2(out_b1)
        out_b3 = self.IMDB2(out_b2)
        out_b4 = self.IMDB2(out_b3)
        # concactate
        out = torch.cat([out_b1, out_b2, out_b3, out_b4], 1)
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out) + out_c1
        # upsample
        out = self.Upsampler(out)

        # bicubic
        #out_bicubic = self.Bilinear(x)

        return out #+ out_bicubic





