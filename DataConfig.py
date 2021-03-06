# -*- coding: utf-8 -*-
# ---
# @File: DataConfig.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe:  CEDAR数据集的读取和划分
# ---
import numpy as np
from itertools import combinations,product
import os
import pickle
from torch.utils.data import Dataset
import cv2
from preprocessing import hafemann_preprocess
import torch



def save_pairs():
    '''
    CDEDAR数据集的划分
    '''
    org_path = r'E:\material\signature\signatures\full_org\original_%d_%d.png'
    forg_path = r'E:\material\signature\signatures\full_forg\forgeries_%d_%d.png'

    forg_author = 55  # num of writers
    org_author = 55
    forg_num = 24  # signatures of each writer
    org_num = 24

    M = 50  # num of writers for training
    K = forg_author - M
    test_writer = np.random.choice(range(1,org_author+1),K,replace=False)
    train_writer = np.arange(1, forg_author + 1)
    train_writer = train_writer[~np.isin(train_writer, test_writer)]
    np.random.shuffle(train_writer)

    pos_pairs = np.array(list(combinations(range(1, org_num + 1), 2)))  # positive pairs, full combinations
    neg_ind=np.random.choice(range(0,org_num**2),pos_pairs.shape[0],replace=False)
    neg_pairs = np.array(list(product(range(1, forg_num + 1),range(1, forg_num + 1))))
    neg_pairs=neg_pairs[neg_ind,:]  # negative pairs,subset of full combinations
    # get pairs
    train_file_ind = []
    for i in train_writer:
        for j in range(pos_pairs.shape[0]):
            positive = [org_path % (i, pos_pairs[j, 0]),org_path % (i, pos_pairs[j, 1]),1]
            negative = [org_path % (i, neg_pairs [j,0]),forg_path % (i, neg_pairs [j,1]),0]
            train_file_ind.append(positive)
            train_file_ind.append(negative)

    train_file_ind=np.array(train_file_ind)


    test_file_ind = []
    for i in test_writer:
        for j in range(pos_pairs.shape[0]):
            positive = [org_path % (i, pos_pairs[j, 0]),org_path % (i, pos_pairs[j, 1]),1]
            negative = [org_path % (i, neg_pairs [j,0]),forg_path % (i, neg_pairs [j,1]),0]
            test_file_ind.append(positive)
            test_file_ind.append(negative)

    test_file_ind=np.array(test_file_ind)

    save_dir = './pair_ind/cedar_ind'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    with open('./pair_ind/cedar_ind/train_index.pkl', 'wb') as train_index_file:
        pickle.dump(train_file_ind, train_index_file)

    with open('./pair_ind/cedar_ind/test_index.pkl', 'wb') as test_index_file:
        pickle.dump(test_file_ind, test_index_file)

    return train_file_ind,test_file_ind

class dataset(Dataset):
    def __init__(self, train=True):
        super(dataset, self).__init__()
        if train:
            path= './pair_ind/cedar_ind/train_index.pkl'
        else:
            path = './pair_ind/cedar_ind/test_index.pkl'

        with open(path, 'rb') as f:
            img_paths = pickle.load(f)
        img_paths=np.array(img_paths)
        self.img_paths = img_paths

    def __len__(self):
        return self.img_paths.shape[0]

    def __getitem__(self, index):
        ind=self.img_paths[index]
        refer, test, label = ind
        refer_img = cv2.imread(refer, 0)
        test_img = cv2.imread(test, 0)
        refer_img = hafemann_preprocess(refer_img,820,890)
        refer_img=np.expand_dims(refer_img,axis=0)

        test_img = hafemann_preprocess(test_img,820,890)
        test_img=np.expand_dims(test_img,axis=0)
        refer_test = np.concatenate((refer_img, test_img), axis=0)
        return torch.FloatTensor(refer_test), float(label)

if __name__=="__main__":
    train_file_ind,test_file_ind=save_pairs()