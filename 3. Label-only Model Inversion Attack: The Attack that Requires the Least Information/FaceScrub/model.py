from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, nc, ndf, nz):
        super(Classifier, self).__init__()

        self.nc = nc
        self.ndf = ndf
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 3, 1, 1),
            nn.BatchNorm2d(ndf),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 3, 1, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 3, 1, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*8) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(ndf * 8 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):

        x = x.view(-1, 1, 64, 64)
        x = self.encoder(x)
        x = x.view(-1, self.ndf * 8 * 4 * 4)
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)
            
            

class Inversion(nn.Module):
    def __init__(self, nc, ngf, nz,truncation, c):
        super(Inversion, self).__init__()
        self.nz = nz
        self.nc = nc
        self.ngf = ngf
        self.truncation = truncation
        self.c = c
        
        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0),
            nn.BatchNorm2d(ngf * 8),
            nn.Tanh(),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1),
            nn.BatchNorm2d(ngf * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.BatchNorm2d(ngf * 2),
            nn.Tanh(),
            # state size. (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.BatchNorm2d(ngf ),
            nn.Tanh(),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

    def truncation_vector(self, x):
        topk, indices = torch.topk(x, self.truncation)
        topk = torch.clamp(torch.log(topk), min=-1000) + self.c
        topk_min = topk.min(1, keepdim=True)[0]
        topk = topk + F.relu(-topk_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, topk)

    def transform_h(self, x):
        x_max, idx = torch.topk(x, 1)
        for i in range(len(x)):
            x_max[i][0] = self.h[idx[i][0]]
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, idx, x_max)
        for i in range(len(x)):
            ave_val = (1 - self.h[idx[i][0]])/(self.nz-1)
            for j in range(self.nz):
                if j != idx[i][0]:
                    x[i][j] = ave_val
        return x

    def one_hot(self, x):
        x_max, idx = torch.topk(x, 1)
        for i in range(len(x)):
            x_max[i][0] = 1
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, idx, x_max)

        return x
          
    def forward(self, x):
    
        if self.truncation == -1:
            # our method
            x = self.transform_h(x)
        elif self.truncation == 0:
            # one hot
            x = self.one_hot(x)
        else:
            # vector-based or score-based
            x = self.truncation_vector(x)
       
        x = x.view(-1, self.nz, 1, 1)
        x = self.decoder(x)
        x = x.view(-1, 1, 64, 64)
        return x

    
# binary classifiers for one specific class
class LRmodule(nn.Module):
    def __init__(self, input_size):
        super(LRmodule, self).__init__()
        self.input_size = input_size
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        outdata = x.view(x.size(0), -1)
        outdata = self.fc(outdata)
        return torch.sigmoid(outdata)



        