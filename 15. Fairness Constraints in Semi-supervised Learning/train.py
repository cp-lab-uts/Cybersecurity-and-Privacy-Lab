from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import pandas as pd
import  csv

import torch
import torch.nn.functional as F
import torch.optim as optim

from utils import load_data, accuracy, load_sampled_data, print2file, demographic_parity, FPR, FNR, OMR
from models import GCN
from Fairness_metrics import Fairness_metrics

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=0.001,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--logDir', type=str, default="Default.txt",
                    help='log file')
parser.add_argument('--fare', type=int, default=1,
                    help='Learning Type.(1 for fare learning')
parser.add_argument('--fair_metric', type=int, default=1,
                    help='fairness metric.(1 for demographic_parity')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='fare learning parameter.')
parser.add_argument('--training_mode', type=float, default=1,
                    help='1 for semi-supervised learning.')
parser.add_argument('--num_unlabel', type=float, default=400,
                   help='the number of unlabeled data.')
parser.add_argument('--num_labeled', type=float, default=1000,
                    help='the number of labeled data.')

args = parser.parse_args()
print2file(str(args), args.logDir, True)
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
adj, features, sens_features, labels, idx_train_label, idx_train_unlabel, idx_val, idx_test = load_sampled_data()

#idx_train = idx_train
num_unlabel = int(args.num_unlabel)
idx_train_unlabel = idx_train_unlabel[:num_unlabel]
idx_train = torch.cat((idx_train_label,idx_train_unlabel),0)


# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)


torch.cuda.empty_cache()

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train_label = idx_train_label.cuda()
    idx_train_unlabel = idx_train_unlabel.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

#import os
#os.system('run.sh')

def train(epoch, idx_train):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    #print('output',output)

    if args.training_mode == 1:
        idx_train = idx_train[:args.num_labeled]


    if args.fare == 1:
        #loss_train = calculate_loss(output[idx_train], sens_features[idx_train], labels[idx_train], alpha=args.const)
        output_exp = torch.exp(output)
        if args.fair_metric == 1:
            const = args.alpha * demographic_parity(sens_features[idx_train], output_exp[idx_train])
        elif args.fair_metric == 2:
            const = args.alpha * FPR(sens_features[idx_train], labels[idx_train], output_exp[idx_train])
        elif args.fair_metric == 3:
            const = args.alpha * FNR(sens_features[idx_train], labels[idx_train], output_exp[idx_train])
        elif args.fair_metric == 4:
            const = args.alpha * OMR(sens_features[idx_train], labels[idx_train], output_exp[idx_train]) #+ args.alpha * FNR(sens_features[idx_train], labels[idx_train], output_exp[idx_train])
        print('const',const)
        loss_train = F.nll_loss((output[idx_train_label]), labels[idx_train_label]) + const
    else:
        loss_train = F.nll_loss((output[idx_train_label]), labels[idx_train_label])
    acc_train = accuracy(output[idx_train], labels[idx_train])

    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss((output[idx_val]), labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    buf = ('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    print2file(buf, args.logDir, True)
    # print('Epoch: {:04d}'.format(epoch+1),
    #       'loss_train: {:.4f}'.format(loss_train.item()),
    #       'acc_train: {:.4f}'.format(acc_train.item()),
    #       'loss_val: {:.4f}'.format(loss_val.item()),
    #       'acc_val: {:.4f}'.format(acc_val.item()),
    #       'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)

    if args.fare == 1:
        loss_test = F.nll_loss((output[idx_test]), labels[idx_test])
    else:
        loss_test = F.nll_loss((output[idx_test]), labels[idx_test])
    predicted = torch.max(output,1)[1]
    acc_test = accuracy(output[idx_test], labels[idx_test])

    buf = ("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))
    print2file(buf, args.logDir, True)
    # print("Test set results:",
    #       "loss= {:.4f}".format(loss_test.item()),
    #       "accuracy= {:.4f}".format(acc_test.item()))
    return predicted[idx_test], acc_test

def calculate_loss(output, sens_features, labels, alpha=10):
    """
    calculate 
    """
    output_exp = torch.exp(output)
    #print('output_exp',torch.exp(output))

    pos_idx = torch.nonzero(sens_features).squeeze(1)
    neg_idx = (sens_features == 0).nonzero().squeeze(1)
    #print('pos_idx',pos_idx)
    # p_i
    pos_pi = output_exp[pos_idx].gather(-1, labels[pos_idx].unsqueeze(1)).squeeze(1)
    neg_pi = output_exp[neg_idx].gather(-1, labels[neg_idx].unsqueeze(1)).squeeze(1)

    pos_dp = torch.sum(pos_pi) / len(pos_idx)
    neg_dp = torch.sum(neg_pi) / len(neg_idx)
    dp = torch.abs(pos_dp - neg_dp)

    pos_fpr = torch.sum(pos_pi * (1 - labels[pos_idx]) / len(pos_idx))
    neg_fpr = torch.sum(neg_pi * (1 - labels[neg_idx]) / len(neg_idx))

    fpr = torch.abs(pos_fpr - neg_fpr)

    pos_fnr = torch.sum((1 - pos_pi) * labels[pos_idx] / len(pos_idx))
    neg_fnr = torch.sum((1 - neg_pi) * labels[neg_idx] / len(neg_idx))

    fnr = torch.abs(pos_fnr - neg_fnr)
    #eo = fpr + fnr

    #loss = F.nll_loss(output, labels) + alpha * (dp + eo)
    classification_loss = F.nll_loss((output), labels)
    loss = classification_loss + alpha * (dp)

    return loss

# Train model
t_total = time.time()

for epoch in range(args.epochs):
    train(epoch, idx_train)

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
print("Start Testing.")
predicted_labels, acc_test = test()
print("End. So far so god...")


# fairness evaluation
test_sens_features = sens_features[idx_test].cpu().numpy()
test_labels = labels[idx_test].cpu().numpy()
test_predicted_labels = predicted_labels.cpu().numpy()
print('number of 0 in the test predicted label', test_predicted_labels.tolist().count(0))
df = pd.DataFrame({'sens_features':test_sens_features,'true_labels':test_labels,'predicted_labels':test_predicted_labels})
fairness_metrics = Fairness_metrics(df,'sens_features','predicted_labels')
acc_test = acc_test.cpu().numpy()

if args.fair_metric == 1:
    dp = fairness_metrics.demographic_parity()
    dis_dp = dp[0]
    f = open("unlabel_titanic_dp","a",)
    #csv_write = csv.writer(f)
    #csv_write.writerow(str(dis_dp))
    f.write(str(dis_dp) + ' ')
    f.write(str(acc_test) + ' ')
    f.close()
    print(dp)
    print('fairness level of dp: {:.4f}'.format(dp[0]))
elif args.fair_metric == 2:
    fpr = fairness_metrics.equal_opportunity('true_labels')
    dis_fpr = fpr[2]
    f = open("titanic_fpr","a")
    f.write(str(dis_fpr) + ' ')
    f.write(str(acc_test) + ' ')
    f.close()
    print('fairness level of fpr',fpr[2],fpr[3],fpr[4])
elif args.fair_metric == 3:
    fnr = fairness_metrics.equal_opportunity('true_labels')
    dis_fnr = fnr[3]
    f = open("titanic_fnr","a")
    f.write(str(dis_fnr) + ' ')
    f.write(str(acc_test) + ' ')
    f.close()
    print('fairness level of fnr: {:.4f}'.format(fnr[3]))
elif args.fair_metric == 4:
    OMR = fairness_metrics.equal_opportunity('true_labels')
    dis_omr = OMR[4]
    f = open("unlabel_titanic_omr","a")
    f.write(str(dis_omr) + ' ')
    f.write(str(acc_test) + ' ')
    f.close()
    print('fairness level of OMR',OMR[4],OMR[2],OMR[3])



#np.savetxt('dp_accuracy',acc_list)
#np.savetxt('dp_discrimination',dis_list)

