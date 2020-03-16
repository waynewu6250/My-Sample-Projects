import torch
import torch.nn as nn
import torch.nn.functional as F

class SegNet(nn.Module):

    def __init__(self, opt, input_dim):

        super(SegNet, self).__init__()

        self.opt = opt

        self.conv1 = nn.Conv2d(input_dim, opt.nChannel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(opt.nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(opt.nConv-1):
            self.conv2.append( nn.Conv2d(opt.nChannel, opt.nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(opt.nChannel) )
        self.conv3 = nn.Conv2d(opt.nChannel, opt.nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(opt.nChannel)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.opt.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x

