from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, nz):
        super(Classifier, self).__init__()

        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 32 x 32
            nn.Conv2d(1, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf) x 16 x 16
            nn.Conv2d(128, 128 * 2, 3, 1, 1),
            nn.BatchNorm2d(128 * 2),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*2) x 8 x 8
            nn.Conv2d(128 * 2, 128 * 4, 3, 1, 1),
            nn.BatchNorm2d(128 * 4),
            nn.MaxPool2d(2, 2, 0),
            nn.ReLU(True),
            # state size. (ndf*4) x 4 x 4
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4 * 4, nz * 5),
            nn.Dropout(0.5),
            nn.Linear(nz * 5, nz),
        )

    def forward(self, x, release=False):

        x = x.view(-1, 1, 32, 32)
        x = self.encoder(x)
        x = x.view(-1, 128 * 4 * 4 * 4)
        x = self.fc(x)

        if release:
            return F.softmax(x, dim=1)
        else:
            return F.log_softmax(x, dim=1)


class Inversion(nn.Module):
    def __init__(self, nz, truncation, c, h):
        super(Inversion, self).__init__()

        self.nz = nz
        self.truncation = truncation
        self.c = c
        self.h = h

        self.decoder = nn.Sequential(
            # input is Z
            nn.ConvTranspose2d(nz, 128 * 4, 4, 1, 0),
            nn.BatchNorm2d(128 * 4),
            nn.Tanh(),
            # state size. (ngf*4) x 4 x 4
            nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1),
            nn.BatchNorm2d(128 * 2),
            nn.Tanh(),
            # state size. (ngf*2) x 8 x 8
            nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            # state size. (ngf) x 16 x 16
            nn.ConvTranspose2d(128, 1, 4, 2, 1),
            nn.Sigmoid()
            # state size. (nc) x 32 x 32
        )

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

    def truncation_vector(self, x):
        top_k, indices = torch.topk(x, self.truncation)
        top_k = torch.clamp(torch.log(top_k), min=-1000) + self.c
        top_k_min = top_k.min(1, keepdim=True)[0]
        top_k = top_k + F.relu(-top_k_min)
        x = torch.zeros(len(x), self.nz).cuda().scatter_(1, indices, top_k)
        # x = torch.zeros(len(x), self.nz).scatter_(1, indices, top_k)

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
        x = x.view(-1, 1, 32, 32)
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