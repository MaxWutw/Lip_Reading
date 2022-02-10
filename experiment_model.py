# import torch
# import torch.nn as nn
# import torch.nn.init as init
# import torch.nn.functional as F
# import math
# import numpy as np
# # from pytorch_pretrained_bert import BertModel
# from transformers import BertModel
# from transformers import AlbertConfig, AlbertModel
# from transformers import AutoModel
#
# class LipNet(torch.nn.Module):
#     def __init__(self, dropout_p=0.5):
#         super(LipNet, self).__init__()
#         self.conv1 = nn.Conv3d(3, 32, (3, 5, 5), (1, 2, 2), (1, 2, 2))
#         self.pool1 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#
#         self.conv2 = nn.Conv3d(32, 64, (3, 5, 5), (1, 1, 1), (1, 2, 2))
#         self.pool2 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#
#         self.conv3 = nn.Conv3d(64, 96, (3, 3, 3), (1, 1, 1), (1, 1, 1))
#         self.pool3 = nn.MaxPool3d((1, 2, 2), (1, 2, 2))
#         # self.FC_reduction = nn.Linear(921600, 3088)
#
#         self.lstm1 = nn.LSTM(96*4*8, 256, 1, bidirectional=True)
#         self.lstm2 = nn.LSTM(512, 256, 1, bidirectional=True)
#
#         self.gru1  = nn.GRU(96*4*8, 256, 1, bidirectional=True)
#         self.gru2  = nn.GRU(512, 256, 1, bidirectional=True)
#         # BertModel = BertModel.from_pretrained('bert-base-uncased')
#         # self.BERT = BertModel(512)
#         # self.BERT = AutoModel.from_pretrained("FPTAI/vibert-base-cased")
#         #albert_xxlarge_configuration = AlbertConfig()
#
#         # self.BERT = AlbertModel.from_pretrained("albert-base-v2")
#
#         self.FC    = nn.Linear(512, 27+1)
#         self.dropout_p  = dropout_p
#
#         self.relu = nn.ReLU(inplace=True)
#         self.dropout = nn.Dropout(self.dropout_p)
#         self.dropout3d = nn.Dropout3d(self.dropout_p)
#         self._init()
#
#     def _init(self):
#
#         init.kaiming_normal_(self.conv1.weight, nonlinearity='relu')
#         init.constant_(self.conv1.bias, 0)
#
#         init.kaiming_normal_(self.conv2.weight, nonlinearity='relu')
#         init.constant_(self.conv2.bias, 0)
#
#         init.kaiming_normal_(self.conv3.weight, nonlinearity='relu')
#         init.constant_(self.conv3.bias, 0)
#
#         init.kaiming_normal_(self.FC.weight, nonlinearity='sigmoid')
#         init.constant_(self.FC.bias, 0)
#
#         for m in (self.gru1, self.gru2):
#             stdv = math.sqrt(2 / (96 * 3 * 6 + 256))
#             for i in range(0, 256 * 3, 256):
#                 init.uniform_(m.weight_ih_l0[i: i + 256],
#                             -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0[i: i + 256])
#                 init.constant_(m.bias_ih_l0[i: i + 256], 0)
#                 init.uniform_(m.weight_ih_l0_reverse[i: i + 256],
#                             -math.sqrt(3) * stdv, math.sqrt(3) * stdv)
#                 init.orthogonal_(m.weight_hh_l0_reverse[i: i + 256])
#                 init.constant_(m.bias_ih_l0_reverse[i: i + 256], 0)
#
#
#     def forward(self, x):
#
#         x = self.conv1(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool1(x)
#
#         x = self.conv2(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool2(x)
#
#         x = self.conv3(x)
#         x = self.relu(x)
#         x = self.dropout3d(x)
#         x = self.pool3(x)
#         # print(x.shape)
#         # (B, C, T, H, W)->(T, B, C, H, W)
#         x = x.permute(2, 0, 1, 3, 4).contiguous()
#         # (B, C, T, H, W)->(T, B, C*H*W)
#         x = x.view(x.size(0), x.size(1), -1)
#         # print(x.shape)
#         # print(x.shape)
#         self.lstm1.flatten_parameters()
#         self.lstm2.flatten_parameters()
#         # x = x.flatten()
#         # x = self.FC_reduction(x)
#         # print(x.unsqueeze(dim=0).shape)
#         x, h = self.lstm1(x)
#         x = self.dropout(x)
#         x, h = self.lstm2(x)
#         print('shape:')
#         print(x.shape)
#         x = self.dropout(x)
#         # x = x.unsqueeze(dim=1).view(16, 193)
#         # x = x.view(16, 193)
#         # x = x.to(torch.int64)
#         # print(x.shape)
#         # x = self.BERT(x)
#         x = self.FC(x)
#         # print(type(x))
#         x = x.permute(1, 0, 2).contiguous()
#         return x
