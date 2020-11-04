import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.downsample = out_channels//in_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=self.downsample, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=self.downsample)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample > 1:
            residual = self.shortcut(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, in_channels, nblocks, fmaps, gran = 1):
        super(ResNet, self).__init__()
        self.fmaps = fmaps
        self.nblocks = nblocks
        self.gran = gran

        self.conv0 = nn.Conv2d(in_channels, fmaps[0], kernel_size=gran*7, stride=gran*2, padding=1)
        self.layer1 = self.block_layers(self.nblocks, [fmaps[0],fmaps[0]])
        self.layer2 = self.block_layers(1, [fmaps[0],fmaps[1]])
        self.layer3 = self.block_layers(self.nblocks, [fmaps[1],fmaps[1]])
        self.fc = nn.Linear(fmaps[1], 1)
        #self.FCN = nn.Sequential(
        #        nn.Linear(fmaps[1], 128), nn.ReLU(), nn.Dropout(p=0.2),
        #        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.2),
        #        nn.Linear(128, 128), nn.ReLU(), nn.Dropout(p=0.2),
        #        nn.Linear(128, 3)
        #        )
        
    def block_layers(self, nblocks, fmaps):
        layers = []
        for _ in range(nblocks):
            layers.append(ResBlock(fmaps[0], fmaps[1]))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.reshape(x, (-1, 8*self.gran, 125*self.gran, 125*self.gran))
        x = self.conv0(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = F.max_pool2d(x, kernel_size=x.size()[2:])
        x = x.view(x.size()[0], self.fmaps[1])
        x = self.fc(x)

        #x = self.FCN(x)
        return x
