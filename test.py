import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import math
import os
import sys
from CMLR_dataset import CMLR
import numpy as np
import time
import torch.optim as optim
import re
import json
from tensorboardX import SummaryWriter
from model import LipNet
from lstm_model import LipNet_lstm
from Convolutional_cat_Transformer import ViTResNet
from Convolutional_cat_Transformer import BasicBlock

opt = __import__('options')
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
torch.set_printoptions(profile="full")
def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
    return DataLoader(dataset,
        batch_size = opt.batch_size,
        shuffle = shuffle,
        num_workers = num_workers,
        drop_last = False)

def show_lr(optimizer):
    lr = []
    for param_group in optimizer.param_groups:
        lr += [param_group['lr']]
    return np.array(lr).mean()

def test(model, net):
    with torch.no_grad():
        dataset = CMLR(opt.video_path,
            opt.anno_path,
            opt.val_list,
            opt.vid_padding,
            opt.txt_padding,
            'test')

        print('num_test_data:{}'.format(len(dataset.data)))
        # model.eval()
        loader = dataset2dataloader(dataset, shuffle=False)
        loss_list = []
        cer = []
        crit = nn.CTCLoss()
        tic = time.time()
        for (i_iter, input) in enumerate(loader):
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            # print(txt.shape) # torch.Size([4, 96, 75, 4, 8])
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            # print(txt)
            # break
            print('test: ', vid.shape)
            # plt.imshow(vid.squeeze(dim=0).permute(0, 3, 2, 1).detach().cpu().numpy())
            y = net(vid.squeeze(dim=0))
            # break
            # y = net(vid)
            # truth = CMLR.model_arr2txt(y)
            # print(truth)
            # print(np.argmax(y))
            # break

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
            loss_list.append(loss)
            pred_txt = [CMLR.model_arr2txt(y.detach().cpu().numpy())]

            truth_txt = [CMLR.arr2txt(txt)]
            cer.extend(CMLR.cer(pred_txt, truth_txt))
            if(i_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(i_iter+1)
                eta = v * (len(loader)-i_iter) / 3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))
                for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
                    print('{:<50}|{:>10}'.format(predict, truth))
                print(''.join(101 *'-'))
                print('test_iter={},eta={},cer={}'.format(i_iter,eta,np.array(cer).mean()))
                print(''.join(101 *'-'))

        return (np.array(loss_list).mean(), np.array(cer).mean())
def train(model, net):

    dataset = CMLR(opt.video_path,
        opt.anno_path,
        opt.train_list,
        opt.vid_padding,
        opt.txt_padding,
        'train')

    loader = dataset2dataloader(dataset)
    optimizer = optim.Adam(model.parameters(),
                lr = opt.base_lr,
                weight_decay = 0.,
                amsgrad = True)
    min_loss = 999999999.
    print('num_train_data:{}'.format(len(dataset.data)))
    crit = nn.CTCLoss()
    tic = time.time()

    train_cer = []
    for epoch in range(opt.max_epoch):
        loss = 0
        cer = 0
        test_loss = 0
        test_cer = 0
        for (i_iter, input) in enumerate(loader):
            model.train()
            vid = input.get('vid').cuda()
            txt = input.get('txt').cuda()
            vid_len = input.get('vid_len').cuda()
            txt_len = input.get('txt_len').cuda()
            optimizer.zero_grad()
            print('train: ', vid.shape)
            y = net(vid.squeeze(dim=0))
            # y = net(vid)

            loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1))
            loss.backward()
            if(opt.is_optimize):
                optimizer.step()

            tot_iter = i_iter + epoch*len(loader)

            pred_txt = [CMLR.model_arr2txt(y.detach().cpu().numpy())]

            truth_txt = [CMLR.arr2txt(txt)]
            train_cer.extend(CMLR.cer(pred_txt, truth_txt))
            cer = CMLR.cer(pred_txt, truth_txt)


            if(tot_iter % opt.display == 0):
                v = 1.0*(time.time()-tic)/(tot_iter+1)
                eta = (len(loader)-i_iter)*v/3600.0

                print(''.join(101*'-'))
                print('{:<50}|{:>50}'.format('predict', 'truth'))
                print(''.join(101*'-'))

                for (predict, truth) in list(zip(pred_txt, truth_txt))[:3]:
                    print('{:<50}|{:>10}'.format(predict, truth))
                print(''.join(101*'-'))
                print('epoch={},tot_iter={},eta={},loss={},train_cer={}'.format(epoch, tot_iter, eta, loss, np.array(train_cer).mean()))
                print(''.join(101*'-'))

            if(tot_iter % opt.test_step == 0):
                (test_loss, test_cer) = test(model, net)
                print('i_iter={},lr={},loss={},cer={}'
                    .format(tot_iter,show_lr(optimizer),test_loss,test_cer))
                if test_loss < min_loss:
                    torch.save(model.state_dict(), '/home/max/Desktop/LipNet-PyTorch-master/weights/current_best_result.pt')
                    min_loss = test_loss
                if(not opt.is_optimize):
                    exit()


        with open('/home/max/Desktop/LipNet-PyTorch-master/training_process_data/gru_train_loss.txt', 'a') as file:
            file.write(str(loss))
            file.write('\n')
        with open('/home/max/Desktop/LipNet-PyTorch-master/training_process_data/gru_train_cer.txt', 'a') as file:
            file.write(str(cer))
            file.write('\n')
        with open('/home/max/Desktop/LipNet-PyTorch-master/training_process_data/gru_test_loss.txt', 'a') as file:
            file.write(str(test_loss))
            file.write('\n')
        with open('/home/max/Desktop/LipNet-PyTorch-master/training_process_data/gru_test_cer.txt', 'a') as file:
            file.write(str(test_cer))
            file.write('\n')

if(__name__ == '__main__'):
    print("Loading options...")
    # model = LipNet()
    model = ViTResNet(BasicBlock, [3, 3, 3])
    # model = LipNet_lstm()
    net = model.cuda()
    # net = nn.DataParallel(model).cuda()

#    if(hasattr(opt, 'weights')):
#         pretrained_dict = torch.load(opt.weights)
#         model_dict = model.state_dict()
#         pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
#         missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
#         print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
#         print('miss matched params:{}'.format(missed_params))
#         model_dict.update(pretrained_dict)
#         model.load_state_dict(model_dict)

    torch.manual_seed(opt.random_seed)
    torch.cuda.manual_seed_all(opt.random_seed)
    train(model, net)
