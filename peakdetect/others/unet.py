import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, n_filters=64, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.bilinear = bilinear
        self.in_c = DoubleConv(self.n_channels, self.n_filters)
        self.down1 = Downscaling(self.n_filters, self.n_filters*2)
        self.down2 = Downscaling(self.n_filters*2, self.n_filters*4)
        self.down3 = Downscaling(self.n_filters*4, self.n_filters*8)
        self.down4 = Downscaling(self.n_filters*8, self.n_filters*8)
        self.up1 = Upscaling(self.n_filters*16, self.n_filters*4, self.bilinear)
        self.up2 = Upscaling(self.n_filters*8, self.n_filters*2, self.bilinear)
        self.up3 = Upscaling(self.n_filters*4, self.n_filters, self.bilinear)
        self.up4 = Upscaling(self.n_filters*2, self.n_filters, self.bilinear)
        self.out_c = OutConv(self.n_filters, self.n_classes)
    
    def forward(self, x):
        x1 = self.in_c(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_c(x)
        return logits

class DoubleConv(nn.Module):
    """[Conv2D -> BN -> ReLU] * 2"""
    def __init__(self, in_channels, out_channels, conv_bias=False):
        super(DoubleConv, self).__init__()
        self.doble_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=conv_bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.doble_conv(x)

class Downscaling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downscaling, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class Upscaling(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upscaling, self).__init__()
        
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        
        self.double_conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffY_div_by_2 = torch.div(diffY, 2, rounding_mode='trunc')
        diffX_div_by_2 = torch.div(diffX, 2, rounding_mode='trunc')
        x1 = F.pad(x1, [diffX_div_by_2, diffX - diffX_div_by_2,
                        diffY_div_by_2, diffY - diffY_div_by_2])

        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
