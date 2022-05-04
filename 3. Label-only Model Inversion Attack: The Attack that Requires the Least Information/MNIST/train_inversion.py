from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import os, shutil
from data import FaceScrub, CelebA, BinaryDataset, extract_dataset
from model import Classifier, Inversion, LRmodule
import torch.nn.functional as F
import torchvision.utils as vutils
import torchvision.datasets as dsets
import logging
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split

import math
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='Adversarial Model Inversion Demo')
parser.add_argument('--batch-size', type=int, default=128, metavar='')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='')
parser.add_argument('--epochs', type=int, default=100, metavar='')
parser.add_argument('--lr', type=float, default=0.01, metavar='')
parser.add_argument('--momentum', type=float, default=0.5, metavar='')
parser.add_argument('--no-cuda', action='store_true', default=False)
parser.add_argument('--seed', type=int, default=1, metavar='')
parser.add_argument('--log-interval', type=int, default=10, metavar='')
# parser.add_argument('--nc', type=int, default=1)
# parser.add_argument('--ndf', type=int, default=128)
# parser.add_argument('--ngf', type=int, default=128)
parser.add_argument('--nz', type=int, default=10)
parser.add_argument('--truncation', type=int, default=10)
parser.add_argument('--c', type=float, default=50.)
parser.add_argument('--num_workers', type=int, default=0, metavar='')
parser.add_argument('--path_out', type=str, default='vector-based/')
parser.add_argument('--early_stop', type=int, default=15)


def train(classifier, inversion, log_interval, device, data_loader, optimizer, epoch, logger):
    classifier.eval()
    inversion.train()

    for batch_idx, (data, target) in enumerate(data_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            prediction = classifier(data, release=True)
        reconstruction = inversion(prediction)
        loss = F.mse_loss(reconstruction, data)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                 len(data_loader.dataset), loss.item()))


def test(classifier, inversion, device, data_loader, epoch, msg, logger, path_out):
    classifier.eval()
    inversion.eval()
    mse_loss = 0
    plot = True
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            prediction = classifier(data, release=True)
            reconstruction = inversion(prediction)
            mse_loss += F.mse_loss(reconstruction, data, reduction='sum').item()

            if plot:
                truth = data[0:512]
                inverse = reconstruction[0:512]
                out = torch.cat((inverse, truth))
                for i in range(16):
                    out[i * 64:i * 64 + 32] = inverse[i * 32:i * 32 + 32]
                    out[i * 64 + 32:i * 64 + 64] = truth[i * 32:i * 32 + 32]
                vutils.save_image(out, path_out + 'recon_{}.png'.format(epoch), nrow=32, normalize=False)
                plot = False

    mse_loss /= len(data_loader.dataset) * 32 * 32
    logger.info('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
    return mse_loss


def main():
    args = parser.parse_args()

    os.makedirs(args.path_out, exist_ok=True)

    logging.basicConfig(level=logging.INFO,
                        filename=args.path_out + 'loss.log',
                        filemode='a',  # 模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                        # a是追加模式，默认果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(pathname)s[line:%(lineno)d]: %(message)s'
                        # 日志格式
                        )
    logger = logging.getLogger(__name__)

    logger.info("================================")
    logger.info(args)
    logger.info("================================")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda == False:
        logger.info('GPU is not used')
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if use_cuda else {}

    torch.manual_seed(args.seed)

    # transform = transforms.Compose([transforms.ToTensor()])

    transform = transforms.Compose([transforms.Resize((32, 32)),
                                    # transforms.Grayscale(num_output_channels=1),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    ])

    train_dataset = dsets.MNIST(root='./data/',
                                train=True,
                                transform=transform,
                                download=True)
    print("len of train_dataset:", len(train_dataset))

    test_dataset = dsets.MNIST(root='./data/',
                               train=False,
                               transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    test_loader_err = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, **kwargs)

    shadow_loader_list = extract_dataset(args.nz, train_dataset, 6000, 5000, 100)
    # tst_dataloader_list = extract_dataset(args.nz, test_set, 50, 50, 1)

    classifier = nn.DataParallel(Classifier(nz=args.nz)).to(device)
    # Load classifier
    path = args.path_out + 'classifier.pth'
    try:
        checkpoint = torch.load(path)

        classifier.load_state_dict(checkpoint['model'])
        # print("test success")
        epoch = checkpoint['epoch']
        best_cl_acc = checkpoint['best_cl_acc']
        print("=> loaded classifier checkpoint '{}' (epoch {}, acc {:.4f})".format(path, epoch, best_cl_acc))
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        return

    posi_num = 0
    neg_num = 0

    classifier.eval()
    for i, (images, labels) in enumerate(test_loader_err):

        images, labels = images.to(device), labels.to(device)
        outputs = classifier(images)

        prob, pred_label = torch.max(outputs, dim=1)

        if pred_label == labels:
            posi_num = posi_num + 1
        else:
            neg_num = neg_num + 1

    print("posi_num:{}, neg_num:{}".format(posi_num, neg_num))
    error_rate = neg_num / (posi_num + neg_num)
    print("target model error_rate: ", error_rate)

    LR_mdl_list = []
    criterion2 = nn.BCELoss()
    for class_name in range(args.nz):
        # LR model and optimizer
        LR_mdl = LRmodule(1 * 32 * 32)
        LR_mdl = LR_mdl.cuda()
        optimizer2 = torch.optim.SGD(LR_mdl.parameters(), lr=0.002)
        for epoch in range(20):
            for i, (images, labels) in enumerate(shadow_loader_list[class_name]):
                images, labels = images.to(device), labels.to(device)
                # Forward + Backward + Optimize
                optimizer2.zero_grad()
                outputs = LR_mdl(images)
                labels = labels.type(torch.float32).unsqueeze(-1)

                loss = criterion2(outputs, labels)
                loss.backward()
                optimizer2.step()

                # if (i + 1) % 20 == 0:
                #   print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                #         % (epoch + 1, num_epochs_train, i + 1, len(trn_dataloader_list[class_name]), loss.data))

        # store trained model
        LR_mdl_list.append(LR_mdl)
    print("---------------------3.Shadow Model Training Finish---------------------")

    import scipy.stats as st
    sigma = 0.75
    med_dist = -sigma * st.norm.ppf(error_rate)
    print("median distance of target model: ", med_dist)
    print("-------------------median distance computation finish--------------------")

    h_value_list = []
    for class_name in range(args.nz):
        w = LR_mdl_list[class_name].fc.weight
        w = w.cpu()
        w_array = w.detach().numpy()
        w_norm = np.linalg.norm(w_array)

        wx_b = med_dist * w_norm

        def sigmoid1(x):
            return 1 / (1 + math.exp(-x))

        h_value = sigmoid1(wx_b)

        h_value_list.append(h_value)

    inversion = nn.DataParallel(Inversion(nz=args.nz, truncation=args.truncation, c=args.c, h=h_value_list)).to(device)
    optimizer = optim.Adam(inversion.parameters(), lr=args.lr, betas=(0.5, 0.999), amsgrad=True)

    # Train inversion model
    best_recon_loss = 999
    early_stop_label = 0
    for epoch in range(1, args.epochs + 1):
        train(classifier, inversion, args.log_interval, device, train_loader, optimizer, epoch, logger)
        recon_loss = test(classifier, inversion, device, test_loader, epoch, 'test1', logger, args.path_out)
        # test(classifier, inversion, device, test2_loader, epoch, 'test2')

        if recon_loss < best_recon_loss:
            best_recon_loss = recon_loss
            state = {
                'epoch': epoch,
                'model': inversion.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_recon_loss': best_recon_loss
            }
            torch.save(state, args.path_out + 'inversion.pth')
            shutil.copyfile(args.path_out + 'recon_{}.png'.format(epoch), args.path_out + 'best.png')
            # shutil.copyfile('out/recon_test2_{}.png'.format(epoch), 'out/best_test2.png')

            early_stop_label = 0
        else:
            early_stop_label += 1
            if early_stop_label == args.early_stop:
                break


if __name__ == '__main__':
    main()