import numpy as np
import scipy.sparse as sp
import torch
import time
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist
from sklearn import preprocessing
import sys
import os
import pickle
import datetime


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def print2file(buf, outFile, p=False):
    if p:
        print(buf)
        outfd = open(outFile, "a+")
        outfd.write(str(datetime.datetime.now()) + "\t" + str(buf) + "\n")
        outfd.close


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))
    start_time = time.time()

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    print("Data Loading Done.  " + str(time.time()-start_time) + "s.")

    return adj, features, labels, idx_train, idx_val, idx_test

def load_sampled_data():
    """
    Load custom sampled data
    """
    def _solve_w(x, sigma):
        dist = squareform(pdist(x[:, 1:], metric='euclidean'))
        f = np.exp(- np.power(dist, 2) / (sigma ** 2))
        return f

    start_time = time.time()
    pickle_name = "titanic_dataset.pk"
    if os.path.isfile(pickle_name):
        print("Loading from pickle.")
        pk = pickle.load(open(pickle_name, "rb"))
    else:
        print("Preprocessing data.")
        pk = {}
        dataset = pd.read_pickle(open(os.path.join(sys.path[0], "titanic_dataset.csv"), "rb"))

        if dataset.shape[1] == 21:
            dataset = dataset[:25010]
            dataset['age'][dataset['age'] > 0] = 1
            dataset['age'][dataset['age'] < 0] = 0
            dataset['labels'][dataset['labels'] > 0] = 1
            dataset['labels'][dataset['labels'] < 0] = 0

            x = np.array(dataset.drop(['labels'], axis=1), dtype=float)
            sens_x = np.array(dataset["age"], dtype=float)
            y = np.array(dataset['labels'], dtype=float)
            adj = sp.coo_matrix(_solve_w(x, 0.1))

            idx_data = range(0, len(x))
            num_train_label = 2000  # int(0.5 * len(x))
            num_train_unlabel = 20000  # int(0.5 * len(x))
            num_val = 10
            num_test = 2000  # int(0.3 * len(x))


        if dataset.shape[1] == 133:
            dataset = dataset[:25000]
            dataset['age'][dataset['age'] > 0] = 1
            dataset['age'][dataset['age'] < 0] = 0
            dataset['labels'][dataset['labels'] > 0] = 1
            dataset['labels'][dataset['labels'] < 0] = 0

            x = np.array(dataset.drop(['labels'], axis=1), dtype=float)
            sens_x = np.array(dataset["age"], dtype=float)
            y = np.array(dataset['labels'], dtype=float)
            adj = sp.coo_matrix(_solve_w(x, 0.1))

            idx_data = range(0, len(x))
            num_train_label = 4000  # int(0.5 * len(x))
            num_train_unlabel = 20000  # int(0.5 * len(x))
            num_val = 1000
            num_test = 5000  # int(0.3 * len(x))


        if dataset.shape[1] == 10:
            print('titanic data preprocessing')
            dataset = dataset.rename(columns={'Survived': 'labels', 'Sex': 'age'})
            #dataset = dataset.reset_index(drop=True)
            #dataset['labels'][dataset['labels'] == 0] = -1
            col = dataset.columns
            dataset = preprocessing.scale(dataset)
            dataset = pd.DataFrame(dataset, columns=col).round(decimals=2)
            dataset['age'][dataset['age'] > 0] = 1
            dataset['age'][dataset['age'] < 0] = 0
            dataset['labels'][dataset['labels'] > 0] = 1
            dataset['labels'][dataset['labels'] < 0] = 0

            x = np.array(dataset.drop(['labels'], axis=1), dtype=float)
            sens_x = np.array(dataset["age"], dtype=float)
            y = np.array(dataset['labels'], dtype=float)
            adj = sp.coo_matrix(_solve_w(x, 0.5))

            idx_data = range(0, len(x))
            num_train_label = 200  # int(0.5 * len(x))
            num_train_unlabel = 400 # int(0.5 * len(x))
            num_val = 1
            num_test = 280  # int(0.3 * len(x))

        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

        #features = normalize(x)
        features = (x)
        # sens_features = normalize(sens_x)
        adj = normalize(adj + sp.eye(adj.shape[0]))



        idx_train_label = idx_data[:num_train_label]
        idx_train_unlabel = idx_data[num_train_label:num_train_label+num_train_unlabel]
        idx_val = idx_data[num_train_label+num_train_unlabel:num_train_label+num_train_unlabel+num_val]
        idx_test = idx_data[-num_test:]

        # features = torch.FloatTensor(np.array(features.todense()))
        pk["features"] = torch.FloatTensor(np.array(features))
        pk["sens_features"] = torch.FloatTensor(sens_x)
        pk["labels"] = torch.LongTensor(y)
        pk["adj"] = sparse_mx_to_torch_sparse_tensor(adj)

        pk["idx_train_label"] = torch.LongTensor(idx_train_label)
        pk["idx_train_unlabel"] = torch.LongTensor(idx_train_unlabel)
        pk["idx_val"] = torch.LongTensor(idx_val)
        pk["idx_test"] = torch.LongTensor(idx_test)

        pickle.dump(pk, open(pickle_name, "wb"), protocol=4)

    print("Data Loading Done.  " + str(time.time()-start_time) + "s.")
    return pk["adj"], pk["features"], pk["sens_features"], pk["labels"], pk["idx_train_label"], pk["idx_train_unlabel"], pk["idx_val"], pk["idx_test"]


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def demographic_parity(sens_features, p):
    '''
    p: the probability matrix with softmax
    '''
    num_sens_1 = torch.zeros(1).cuda()
    num_sens_0 = torch.zeros(1).cuda()
    p_1 = torch.zeros(1).cuda()
    p_0 = torch.zeros(1).cuda()

    for i in range(sens_features.size()[0]):
        if sens_features[i] > 0:
            num_sens_1 += torch.tensor(1)
            p_1 += p[i][0]
        else:
            num_sens_0 += torch.tensor(1)
            p_0 += p[i][0]
    const = abs(((p_1 / num_sens_1) - (p_0 / num_sens_0)))
    return const

def FPR(sens_features, labels, p):
    '''
    p: the probability matrix with softmax
    '''
    num_sens_1 = torch.zeros(1).cuda()
    num_sens_0 = torch.zeros(1).cuda()
    p_1 = torch.zeros(1).cuda()
    p_0 = torch.zeros(1).cuda()

    for i in range(sens_features.size()[0]):
        if sens_features[i] > 0:
            num_sens_1 += torch.tensor(1)
            p_1 += p[i][0] * (1-labels[i])
        else:
            num_sens_0 += torch.tensor(1)
            p_0 += p[i][0] * (1-labels[i])
    const = abs(((p_1 / num_sens_1) - (p_0 / num_sens_0)))
    return const

def FNR(sens_features, labels, p):
    '''
    p: the probability matrix with softmax
    '''
    num_sens_1 = torch.zeros(1).cuda()
    num_sens_0 = torch.zeros(1).cuda()
    p_1 = torch.zeros(1).cuda()
    p_0 = torch.zeros(1).cuda()

    for i in range(sens_features.size()[0]):
        if sens_features[i] > 0:
            num_sens_1 += torch.tensor(1)
            p_1 += ((1-p[i][0]) * (labels[i]))
        else:
            num_sens_0 += torch.tensor(1)
            p_0 += ((1-p[i][0]) * (labels[i]))
    const = abs(((p_1 / num_sens_1) - (p_0 / num_sens_0)))
    return const

def OMR(sens_features, labels, p):
    '''
    p: the probability matrix with softmax
    '''
    num_sens_1 = torch.zeros(1).cuda()
    num_sens_0 = torch.zeros(1).cuda()
    p_1 = torch.zeros(1).cuda()
    p_0 = torch.zeros(1).cuda()

    for i in range(sens_features.size()[0]):
        if sens_features[i] > 0:
            num_sens_1 += torch.tensor(1)
            p_1 += (p[i][0] * (1-labels[i]) + (1-p[i][0]) * labels[i])
        else:
            num_sens_0 += torch.tensor(1)
            p_0 += (p[i][0] * (1-labels[i]) + (1-p[i][0]) * labels[i])
    const = abs(((p_1 / num_sens_1) - (p_0 / num_sens_0)))
    return const

def OMR(sens_features, labels, p):
    '''
    p: the probability matrix with softmax
    '''
    num_sens_1 = torch.zeros(1).cuda()
    num_sens_0 = torch.zeros(1).cuda()
    p_1 = torch.zeros(1).cuda()
    p_0 = torch.zeros(1).cuda()

    for i in range(sens_features.size()[0]):
        if sens_features[i] > 0:
            num_sens_1 += torch.tensor(1)
            if p[i][0] > 0.5:

                p_1 += (p[i][0] * (1-labels[i]) + (1-p[i][0]) * labels[i])
        else:
            num_sens_0 += torch.tensor(1)
            p_0 += (p[i][0] * (1-labels[i]) + (1-p[i][0]) * labels[i])
    const = abs(((p_1 / num_sens_1) - (p_0 / num_sens_0)))
    return const
