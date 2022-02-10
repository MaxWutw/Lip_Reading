# encoding: utf-8
import numpy as np
import glob
import time
import cv2
import os
from torch.utils.data import Dataset
from cvtransforms import *
import torch
import glob
import re
import copy
import json
import random
import editdistance
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from PIL import Image


class CMLR(Dataset):
    root_dir = '/home/max/Desktop/annotation/s1'
    root_dir2 = '/home/max/Desktop/annotation/s3'
    root_dir3 = '/home/max/Desktop/annotation/s4'
    root_file = os.listdir(root_dir)
    root_file2 = os.listdir(root_dir2)
    root_file3 = os.listdir(root_dir3)
    word_list = [' ']
    for i in root_file:
        if str(i) == '._.DS_Store' or str(i) == '.DS_Store':
            continue
        child_file = os.path.join(root_dir, i)
        iter_child = os.listdir(child_file)
        for idx in iter_child:
            iter_file = os.path.join(root_dir, i)
            iter_file = iter_file + '/' + str(idx)
            with open(iter_file, 'r') as file:
                lines = [line.strip().split(' ') for line in file.readlines()]
                txt = lines[0]
                txt = "".join(txt)
                for t in txt:
                    if t in word_list:
                        continue
                    word_list.append(t)

    for i in root_file2:
        if str(i) == '._.DS_Store' or str(i) == '.DS_Store':
            continue
        child_file = os.path.join(root_dir2, i)
        iter_child = os.listdir(child_file)
        for idx in iter_child:
            iter_file = os.path.join(root_dir2, i)
            iter_file = iter_file + '/' + str(idx)
            with open(iter_file, 'r') as file:
                lines = [line.strip().split(' ') for line in file.readlines()]
                txt = lines[0]
                txt = "".join(txt)
                for t in txt:
                    if t in word_list:
                        continue
                    word_list.append(t)

    # for i in root_file3:
    #     if str(i) == '._.DS_Store' or str(i) == '.DS_Store':
    #         continue
    #     child_file = os.path.join(root_dir3, i)
    #     iter_child = os.listdir(child_file)
    #     for idx in iter_child:
    #         iter_file = os.path.join(root_dir3, i)
    #         iter_file = iter_file + '/' + str(idx)
    #         with open(iter_file, 'r') as file:
    #             lines = [line.strip().split(' ') for line in file.readlines()]
    #             txt = lines[0]
    #             txt = "".join(txt)
    #             for t in txt:
    #                 if t in word_list:
    #                     continue
    #                 word_list.append(t)

    values = np.array(word_list)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    # foo = np.array([[0]])
    # integer_encoded = np.concatenate([foo, integer_encoded])
    # print(out)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print('one hot: ', len(onehot_encoded))

    def __init__(self, video_path, anno_path, file_list, vid_pad, txt_pad, phase):
        self.anno_path = anno_path
        self.vid_pad = vid_pad
        self.txt_pad = txt_pad
        self.phase = phase

        with open(file_list, 'r') as f:
            # print(f.readlines())
            self.videos = [os.path.join(video_path, line.strip()) for line in f.readlines()]
        self.data = []
        for vid in self.videos:
            items = vid.split(os.path.sep)
            item = items[5] + '/' + items[6] + '/' + items[7]
            # print('item: ', items)
            self.data.append((vid, items[-4], item))

    def __getitem__(self, idx):
        (vid, spk, name) = self.data[idx]
        vid = self._load_vid(vid)
        anno = self._load_anno(os.path.join(self.anno_path, name + '.txt'))
        # for i in vid:
        #     # print(i.shape)
        #     # plt.imshow(i)
        #     # plt.show()
        #     img = Image.fromarray(i)
        #     img.save('my.png')
        #     img.show()
        #     break
        if(self.phase == 'train'):
            vid = HorizontalFlip(vid)

        vid = ColorNormalize(vid)

        vid_len = vid.shape[0]
        anno_len = anno.shape[0]
        # print('shape: ', vid.shape)
        vid = self._padding(vid, self.vid_pad)
        anno = self._padding(anno, self.txt_pad)
## 0312 3012
        return {'vid': torch.FloatTensor(vid.transpose(0, 3, 1, 2)),
            'txt': torch.LongTensor(anno),
            'txt_len': anno_len,
            'vid_len': vid_len}

    def __len__(self):
        return len(self.data)

    def _load_vid(self, p):
        files = os.listdir(p)
        files = list(filter(lambda file: file.find('.jpg') != -1, files))
        files = sorted(files, key=lambda file: int(os.path.splitext(file.split('_')[1])[0]))
        array = [cv2.imread(os.path.join(p, file)) for file in files]

        array = list(filter(lambda im: not im is None, array))
        array = [cv2.resize(im, (32, 32), interpolation=cv2.INTER_LANCZOS4) for im in array]
        # print('array: ', lenarray)
        # array = array.astype(np.float32)
        array = np.stack(array, axis=0).astype(np.float32)
        return array

    def _load_anno(self, name):
        with open(name, 'r') as f:
            lines = [line.strip().split(' ') for line in f.readlines()]
            txt = lines[0]
        return self.txt2arr(' '.join(txt), 1)

    def txt2arr(self, txt, start):
        word_list = []
        output = []
        for t in txt:
            if t in word_list:
                continue
            word_list.append(t)
        word = np.array(word_list)
        data = CMLR.label_encoder.transform(word)
        # data = data.reshape(len(data), 1)
        # data = CMLR.onehot_encoder.transform(data)
        return data

    def arr2txt(arr):
        # print(arr.shape)
        txt = ''
        for i in arr[0]:
            # print(i)
            # break
            inverted = CMLR.label_encoder.inverse_transform([i.cpu()])
            txt += str(inverted[0])
        # print(txt)
        return ''.join(txt).strip()

    def model_arr2txt(arr):
        # print(arr.shape)
        txt = ''
        for i in arr[0]:
            # print(i)
            # break
            # print(type(i))
            # print(np.argmax(i))
            inverted = CMLR.label_encoder.inverse_transform([i.argmax(-1)])
            txt += str(inverted[0])
        # print(txt)
        return ''.join(txt).strip()


    def _padding(self, array, length):
        array = [array[_] for _ in range(array.shape[0])]
        size = array[0].shape
        for i in range(length - len(array)):
            array.append(np.zeros(size))
        return np.stack(array, axis=0)

    @staticmethod
    def wer(predict, truth):
        word_pairs = [(p[0].split(' '), p[1].split(' ')) for p in zip(predict, truth)]
        wer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in word_pairs]
        return wer

    @staticmethod
    def cer(predict, truth):
        cer = [1.0*editdistance.eval(p[0], p[1])/len(p[1]) for p in zip(predict, truth)]
        return cer
