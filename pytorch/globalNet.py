import torch.nn as nn
import torch
import math

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample 
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class globalNet(nn.Module):
    def __init__(self, channel_settings,  num_class):
        super(globalNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict1,downsamples = [], [], [],[]
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i],i))
            predict1.append(self._predict1(num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
                downsamples.append(self._downsample())
        self.fc1=nn.Linear(256*4, num_class)
    
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict1 = nn.ModuleList(predict1)
        self.downsamples=nn.ModuleList(downsamples)
    

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
    
    def _lateral(self, input_size, index):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(SELayer(256))
        layers.append(nn.ReLU(inplace=True))
        for i in range(index+1):
            layers.append(Bottleneck(256,64))

        return nn.Sequential(*layers)


    def _upsample(self):
        layers = []
        layers.append(torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(SELayer(256))

        return nn.Sequential(*layers)

    def _downsample(self):
        layers = []
        layers.append(torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        layers.append(torch.nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(SELayer(256))

        return nn.Sequential(*layers)

    def _predict1(self, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(SELayer(256))
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.AdaptiveAvgPool2d(1))
        return nn.Sequential(*layers)
    

    def forward(self, x):
        global_fms, global_outs = [], []
        features=[[],[],[],[]]
        up=[[],[],[],[]]
        down=[[],[],[],[]]
        for i in range(len(self.channel_settings)):           
            features[i] = self.laterals[i](x[i])
            #print('feature{}:{}'.format(i,features[i].size()))           
            if i != len(self.channel_settings) - 1:
                feature1=features[i]
                for k in range(len(self.channel_settings)-i-1):                   
                    feature1=self.upsamples[k](feature1)
                    up[i].append(feature1)
            if i !=0:
                feature2=features[i]
                for k in range(i):
                    feature2=self.downsamples[i-k-1](feature2)
                    down[i].append(feature2)
        features[0]=features[0]+down[3][2]+down[2][1]+down[1][0]
        features[1]=features[1]+up[0][0]+down[3][1]+down[2][0]
        features[2]=features[2]+up[0][1]+up[1][0]+down[3][0]
        features[3]=features[3]+up[0][2]+up[1][1]+up[2][0]

        for i in range(4):
            features[i]=self.predict1[i](features[i])
            global_fms.append(features[i])    
        out = torch.cat(global_fms, dim=1) 
        out = out.view(out.size(0), -1)
        out=self.fc1(out)
        return out


