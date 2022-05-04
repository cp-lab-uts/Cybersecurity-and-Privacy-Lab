from __future__ import print_function, division
import numpy as np
import torch.utils.data as data
from PIL import Image

class BinaryDataset(data.Dataset):
    def __init__(self, class_name, dataset_name, positive_num=6000, negative_num=5000):
        imgs = []
        num1 = 0
        num0 = 0
        for sample in dataset_name:
            if num0 == -1 and num1 == -1:
                break
            if sample[1] == class_name:
                if num1 >= positive_num or num1 == -1:
                    num1 = -1
                    continue
                labeli = 1
                num1 = num1 + 1
                imgs.append((sample[0], labeli))
            else:
                if num0 >= negative_num or num0 == -1:
                    num0 = -1
                    continue
                labeli = 0
                num0 = num0 + 1
                imgs.append((sample[0], labeli))
        self.imgs = imgs

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return len(self.imgs)


def extract_dataset(class_num, dataset_name, posi_num, nega_num, batch_size):
    data_loader = []
    kwargs1 = {'num_workers': 0, 'pin_memory': True}
    for i in range(class_num):
        ds_i = BinaryDataset(class_name=i, dataset_name=dataset_name, positive_num=posi_num, negative_num=nega_num)
        data_loader_i = data.DataLoader(dataset=ds_i,
                                        batch_size=batch_size,
                                        shuffle=True,
                                        **kwargs1)
        data_loader.append(data_loader_i)
    return data_loader


class ExtractClasses(data.Dataset):
    def __init__(self, dataset_name, class_list):
        imgs = []
        num = 0
        label_dic = {}
        for labeli in class_list:
            new_label = class_list.index(labeli)
            label_dic[labeli] = new_label
        for sample in dataset_name:
            if sample[1] in class_list:
                new_label = label_dic[sample[1]]
                imgs.append((sample[0], new_label))
                num += 1
        self.imgs = imgs
        self.num = num

    def __getitem__(self, index):
        fn = self.imgs[index]
        return fn

    def __len__(self):
        return self.num