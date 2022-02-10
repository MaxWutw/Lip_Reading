# from model import LipNet
#
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.transforms as transforms
#
#
# # class Net(nn.Module):
# #     def __init__(self):
# #         super(Net, self).__init__()
# #         self.conv1 = nn.Conv2d(3, 6, 5)
# #         self.pool1 = nn.MaxPool2d(2, 2)
# #         self.conv2 = nn.Conv2d(6, 16, 5)
# #         self.pool2 = nn.MaxPool2d(2, 2)
# #         self.fc1 = nn.Linear(16 * 5 * 5, 120)
# #         self.fc2 = nn.Linear(120, 84)
# #         self.fc3 = nn.Linear(84, 10)
# #
# #     def forward(self, x):
# #         x = self.pool1(F.relu(self.conv1(x)))
# #         x = self.pool1(F.relu(self.conv2(x)))
# #         x = x.view(-1, 16 * 5 * 5)
# #         x = F.relu(self.fc1(x))
# #         x = F.relu(self.fc2(x))
# #         x = self.fc3(x)
# #         return x
#
#
# def img_transform(img_in, transform):
#     """
#     将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
#     :param img_roi: np.array
#     :return:
#     """
#     img = img_in.copy()
#     img = Image.fromarray(np.uint8(img))
#     img = transform(img)
#     img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
#     return img
#
#
# def img_preprocess(img_in):
#     """
#     读取图片，转为模型可读的形式
#     :param img_in: ndarray, [H, W, C]
#     :return: PIL.image
#     """
#     img = img_in.copy()
#     img = cv2.resize(img,(128, 64))
#     # img = img[:, :, ::-1]   # BGR --> RGB
#     # transform = transforms.Compose([
#     #     transforms.ToTensor(),
#     #     # transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
#     # ])
#     # img_input = img_transform(img, transform)
#     img_input = torch.FloatTensor(img.transpose(3, 0, 1, 2))
#     # print(type(img_input))
#     return img_input
#
#
# def backward_hook(module, grad_in, grad_out):
#     grad_block.append(grad_out[0].detach())
#
#
# def farward_hook(module, input, output):
#     fmap_block.append(output)
#
#
# def show_cam_on_image(img, mask, out_dir):
#     heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
#     heatmap = np.float32(heatmap) / 255
#     cam = heatmap + np.float32(img)
#     cam = cam / np.max(cam)
#
#     path_cam_img = os.path.join(out_dir, "cam.jpg")
#     path_raw_img = os.path.join(out_dir, "raw.jpg")
#     if not os.path.exists(out_dir):
#         os.makedirs(out_dir)
#     cv2.imwrite(path_cam_img, np.uint8(255 * cam))
#     cv2.imwrite(path_raw_img, np.uint8(255 * img))
#
#
# def comp_class_vec(ouput_vec, index=None):
#     """
#     计算类向量
#     :param ouput_vec: tensor
#     :param index: int，指定类别
#     :return: tensor
#     """
#     if not index:
#         index = np.argmax(ouput_vec.cpu().data.numpy())
#     else:
#         index = np.array(index)
#     index = index[np.newaxis, np.newaxis]
#     index = torch.from_numpy(index)
#     one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
#     one_hot.requires_grad = True
#     class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605
#
#     return class_vec
#
#
# def gen_cam(feature_map, grads):
#     """
#     依据梯度和特征图，生成cam
#     :param feature_map: np.array， in [C, H, W]
#     :param grads: np.array， in [C, H, W]
#     :return: np.array, [H, W]
#     """
#     cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)
#
#     weights = np.mean(grads, axis=(1, 2))  #
#
#     for i, w in enumerate(weights):
#         cam += w * feature_map[i, :, :]
#
#     cam = np.maximum(cam, 0)
#     cam = cv2.resize(cam, (32, 32))
#     cam -= np.min(cam)
#     cam /= np.max(cam)
#
#     return cam
#
# def _load_vid(p):
#     files = os.listdir(p)
#     files = list(filter(lambda file: file.find('.jpg') != -1, files))
#     files = sorted(files, key=lambda file: int(os.path.splitext(file.split('_')[1])[0]))
#     array = [cv2.imread(os.path.join(p, file)) for file in files]
#
#     array = list(filter(lambda im: not im is None, array))
#     array = [cv2.resize(im, (128, 64), interpolation=cv2.INTER_LANCZOS4) for im in array]
#     # print('array: ', lenarray)
#     # array = array.astype(np.float32)
#     array = np.stack(array, axis=0).astype(np.float32)
#     return array
#
#
# if __name__ == '__main__':
#
#     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#     print(BASE_DIR)
#     path_img = os.path.join('/home/max/Desktop/video/s3/bwat3a/frame_3.jpg')
#     path_net = os.path.join('/home/max/Downloads/current_best_loss(2).pt')
#     output_dir = os.path.join('/home/max/Desktop/LipNet-PyTorch-master')
#
#     classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#     fmap_block = list()
#     grad_block = list()
#
#     # 图片读取；网络加载
#     # img = cv2.imread(path_img, 1)  # H*W*C
#     # img_input = img_preprocess(img)
#     # print('shape: ', img_input.shape)
#     # img_input = np.array(cv2.resize(img, (128, 64), interpolation=cv2.INTER_LANCZOS4))
#
#     vid = _load_vid('/home/max/Desktop/tmp_folder/bwat3a/')
#     img_input = torch.FloatTensor(vid.transpose(3, 0, 1, 2))
#     print(img_input.shape)
#     net = LipNet()
#     net.load_state_dict(torch.load(path_net))
#
#     # 注册hook
#     net.conv3.register_forward_hook(farward_hook)
#     net.conv3.register_backward_hook(backward_hook)
#
#     # forward
#     output = net(img_input.unsqueeze(dim=0))
#     idx = np.argmax(output.cpu().data.numpy())
#     # print("predict: {}".format(classes[idx]))
#
#     # backward
#     net.zero_grad()
#     # class_loss = comp_class_vec(output)
#     # class_loss.backward()
#
#     # 生成cam
#     grads_val = grad_block[0].cpu().data.numpy().squeeze()
#     fmap = fmap_block[0].cpu().data.numpy().squeeze()
#     cam = gen_cam(fmap, grads_val)
#
#     # 保存cam图片
#     img_show = np.float32(cv2.resize(img, (32, 32))) / 255
#     show_cam_on_image(img_show, cam, output_dir)
#
#
# # import torch
# # import torch.nn as nn
# # import torch.nn.init as init
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # import math
# # import os
# # import sys
# # # from dataset import MyDataset
# # # from MyDataset_dataset import MyDataset
# # import numpy as np
# # import time
# # from model import LipNet
# # import torch.optim as optim
# # import re
# # import json
# # from tensorboardX import SummaryWriter
# # # from Convolutional_cat_Transformer import ViTResNet
# # # from Convolutional_cat_Transformer import BasicBlock
# # from dataset import MyDataset
# #
# # if(__name__ == '__main__'):
# #     opt = __import__('options')
# #     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
# #     writer = SummaryWriter()
# #
# #
# # if(__name__ == '__main__'):
# #     print("Loading options...")
# #     model = LipNet()
# #     # model = ViTResNet(BasicBlock, [3, 3, 3])
# #     net = model.cuda()
# #     # net = nn.DataParallel(model).cuda()
# #
# # #    if(hasattr(opt, 'weights')):
# # # /home/max/Downloads/current_best_loss(2).pt
# #     pretrained_dict = torch.load('/home/max/Downloads/current_best_loss(2).pt')
# #     model_dict = model.state_dict()
# #     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys() and v.size() == model_dict[k].size()}
# #     missed_params = [k for k, v in model_dict.items() if not k in pretrained_dict.keys()]
# #     print('loaded params/tot params:{}/{}'.format(len(pretrained_dict),len(model_dict)))
# #     print('miss matched params:{}'.format(missed_params))
# #     model_dict.update(pretrained_dict)
# #     model.load_state_dict(model_dict)
# #
# #     torch.manual_seed(opt.random_seed)
# #     torch.cuda.manual_seed_all(opt.random_seed)
# #     # train(model, net)
# #
# # def dataset2dataloader(dataset, num_workers=opt.num_workers, shuffle=True):
# #     return DataLoader(dataset,
# #         batch_size = opt.batch_size,
# #         shuffle = shuffle,
# #         num_workers = num_workers,
# #         drop_last = False)
# #
# # def show_lr(optimizer):
# #     lr = []
# #     for param_group in optimizer.param_groups:
# #         lr += [param_group['lr']]
# #     return np.array(lr).mean()
# #
# # def ctc_decode(y):
# #     result = []
# #     y = y.argmax(-1)
# #     return [MyDataset.ctc_arr2txt(y[_], start=1) for _ in range(y.size(0))]
# #
# # with torch.no_grad():
# #     dataset = MyDataset(opt.video_path,
# #         opt.anno_path,
# #         opt.train_list,
# #         opt.vid_padding,
# #         opt.txt_padding,
# #         'train')
# #
# #     print('num_test_data:{}'.format(len(dataset.data)))
# #     model.eval()
# #     loader = dataset2dataloader(dataset, shuffle=False)
# #     loss_list = []
# #     wer = []
# #     cer = []
# #     crit = nn.CTCLoss()
# #     tic = time.time()
# #     for (i_iter, input) in enumerate(loader):
# #         vid = input.get('vid').cuda()
# #         txt = input.get('txt').cuda()
# #         # print(txt.shape) # torch.Size([4, 96, 75, 4, 8])
# #         vid_len = input.get('vid_len').cuda()
# #         txt_len = input.get('txt_len').cuda()
# #         y = net(vid)
# #         # y = net(vid)
# #         # print(y.shape)
# #
# #         loss = crit(y.transpose(0, 1).log_softmax(-1), txt, vid_len.view(-1), txt_len.view(-1)).detach().cpu().numpy()
# #         loss_list.append(loss)
# #         pred_txt = ctc_decode(y)
# #
# #         truth_txt = [MyDataset.arr2txt(txt[_], start=1) for _ in range(txt.size(0))]
# #         wer.extend(MyDataset.wer(pred_txt, truth_txt))
# #         cer.extend(MyDataset.cer(pred_txt, truth_txt))
# #         if(i_iter % opt.display == 0):
# #             v = 1.0*(time.time()-tic)/(i_iter+1)
# #             eta = v * (len(loader)-i_iter) / 3600.0
# #
# #             print(''.join(101*'-'))
# #             print('{:<50}|{:>50}'.format('predict', 'truth'))
# #             print(''.join(101*'-'))
# #             for (predict, truth) in list(zip(pred_txt, truth_txt))[:10]:
# #                 print('{:<50}|{:>50}'.format(predict, truth))
# #             print(''.join(101 *'-'))
# #             print('test_iter={},eta={},wer={},cer={}'.format(i_iter,eta,np.array(wer).mean(),np.array(cer).mean()))
# #             print(''.join(101 *'-'))
# #             # break


import cv2
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from model import LipNet

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16 * 13 * 29, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool1(F.relu(self.conv2(x)))
        print(x.shape)
        x = x.view(-1, 16 * 13 * 29)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def img_transform(img_in, transform):
    """
    将img进行预处理，并转换成模型输入所需的形式—— B*C*H*W
    :param img_roi: np.array
    :return:
    """
    img = img_in.copy()
    img = Image.fromarray(np.uint8(img))
    img = transform(img)
    img = img.unsqueeze(0)    # C*H*W --> B*C*H*W
    return img


def img_preprocess(img_in):
    """
    读取图片，转为模型可读的形式
    :param img_in: ndarray, [H, W, C]
    :return: PIL.image
    """
    img = img_in.copy()
    img = cv2.resize(img,(128, 64))
    img = img[:, :, ::-1]   # BGR --> RGB
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4948052, 0.48568845, 0.44682974], [0.24580306, 0.24236229, 0.2603115])
    ])
    img_input = img_transform(img, transform)
    return img_input


def backward_hook(module, grad_in, grad_out):
    grad_block.append(grad_out[0].detach())


def farward_hook(module, input, output):
    fmap_block.append(output)


def show_cam_on_image(img, mask, out_dir, i):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, f"cam_{i}.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))


def comp_class_vec(ouput_vec, index=None):
    """
    计算类向量
    :param ouput_vec: tensor
    :param index: int，指定类别
    :return: tensor
    """
    if not index:
        index = np.argmax(ouput_vec.cpu().data.numpy())
    else:
        index = np.array(index)
    index = index[np.newaxis, np.newaxis]
    index = torch.from_numpy(index)
    one_hot = torch.zeros(1, 10).scatter_(1, index, 1)
    one_hot.requires_grad = True
    class_vec = torch.sum(one_hot * output)  # one_hot = 11.8605

    return class_vec


def gen_cam(feature_map, grads):
    """
    依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # cam shape (H, W)

    weights = np.mean(grads, axis=(1, 2))  #

    for i, w in enumerate(weights):
        cam += w * feature_map[i, :, :]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (128, 64))
    cam -= np.min(cam)
    cam /= np.max(cam)

    return cam


if __name__ == '__main__':
# for i in range(75):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    path_img = os.path.join(f'/home/max/Desktop/video/s3/bwat3a/frame_65.jpg')
    path_net = os.path.join('/home/max/Downloads/current_best_loss(2).pt')
    output_dir = os.path.join('/home/max/Desktop/LipNet-PyTorch-master')

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    fmap_block = list()
    grad_block = list()

    # 图片读取；网络加载
    img = cv2.imread(path_img, 1)  # H*W*C
    img_input = img_preprocess(img)
    net = Net()
    # net.load_state_dict(torch.load(path_net))

    # 注册hook
    net.conv2.register_forward_hook(farward_hook)
    net.conv2.register_backward_hook(backward_hook)

    # forward
    output = net(img_input)
    idx = np.argmax(output.cpu().data.numpy())
    print("predict: {}".format(classes[idx]))

    # backward
    net.zero_grad()
    class_loss = comp_class_vec(output)
    class_loss.backward()

    # 生成cam
    grads_val = grad_block[0].cpu().data.numpy().squeeze()
    fmap = fmap_block[0].cpu().data.numpy().squeeze()
    cam = gen_cam(fmap, grads_val)

    # 保存cam图片
    img_show = np.float32(cv2.resize(img, (128, 64))) / 255
    show_cam_on_image(img_show, cam, output_dir, 65)
