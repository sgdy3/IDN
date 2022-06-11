# -*- coding: utf-8 -*-
# ---
# @File: texture_mat.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/3/22
# Describe: 实现了signet-f
# ---



import torch
import torch.nn as nn
import numpy as np



class StreamThin(nn.Module):
    def __init__(self):
        super(StreamThin, self).__init__()

        self.stream = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 48, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(48, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.Conv_16 = nn.Conv2d(16, 16, (3, 3), stride=(1, 1), padding=1)
        self.Conv_32 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.Conv_48 = nn.Conv2d(48, 48, (3, 3), stride=(1, 1), padding=1)
        self.Conv_64 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.Conv_96 = nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1)

        self.fc_16 = nn.Linear(16, 16)
        self.fc_32 = nn.Linear(32, 32)
        self.fc_48 = nn.Linear(48, 48)
        self.fc_64 = nn.Linear(64, 64)
        self.fc_128 = nn.Linear(128, 128)


        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, reference, inverse):
        for i in range(4):
            reference = self.stream[0 + i * 5](reference)
            reference = self.stream[1 + i * 5](reference)
            inverse = self.stream[0 + i * 5](inverse)
            inverse = self.stream[1 + i * 5](inverse)
            inverse = self.stream[2 + i * 5](inverse)
            inverse = self.stream[3 + i * 5](inverse)
            inverse = self.stream[4 + i * 5](inverse)
            reference = self.attention(inverse, reference)
            reference = self.stream[2 + i * 5](reference)
            reference = self.stream[3 + i * 5](reference)
            reference = self.stream[4 + i * 5](reference)

        return reference, inverse

    def attention(self, inverse, discrimnative):
        """
        注意力模块的设置原理：
        1. 抽出反色流的第二层卷积结果，上采样到和判决流第一层卷积结果相同大小；
        2. 过一层conv提取特征
        3. 与判决流第一层卷积结果做内积再相加，类似skip connection
        4. 分出两路，一路过GAP过FC后再reszie到原大小，成为注意力
           注意力和另一路相乘
        """
        GAP = nn.AdaptiveAvgPool2d((1, 1))
        sigmoid = nn.Sigmoid()
        up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
        conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
        g = conv(up_sample)
        g = sigmoid(g)
        tmp = g * discrimnative + discrimnative
        f = GAP(tmp)
        f = f.view(f.size()[0], 1, f.size()[1])
        fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
        f = fc(f)
        f = sigmoid(f)
        f = f.view(-1, f.size()[2], 1, 1)
        out = tmp * f
        return out


class StreamStandard(nn.Module):
    def __init__(self):
        super(StreamStandard, self).__init__()

        self.stream = nn.Sequential(
            nn.Conv2d(1, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(32, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 96, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(96, 128, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, (3,3), stride=(1,1), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        self.Conv_32 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=1)
        self.Conv_64 = nn.Conv2d(64, 64, (3, 3), stride=(1, 1), padding=1)
        self.Conv_96 = nn.Conv2d(96, 96, (3, 3), stride=(1, 1), padding=1)
        self.Conv_48 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=1)


        self.fc_32 = nn.Linear(32, 32)
        self.fc_64 = nn.Linear(64, 64)
        self.fc_96 = nn.Linear(96, 96)
        self.fc_128 = nn.Linear(128, 128)


        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, reference, inverse):
        for i in range(4):
            reference = self.stream[0 + i * 5](reference)
            reference = self.stream[1 + i * 5](reference)
            inverse = self.stream[0 + i * 5](inverse)
            inverse = self.stream[1 + i * 5](inverse)
            inverse = self.stream[2 + i * 5](inverse)
            inverse = self.stream[3 + i * 5](inverse)
            inverse = self.stream[4 + i * 5](inverse)
            reference = self.attention(inverse, reference)
            reference = self.stream[2 + i * 5](reference)
            reference = self.stream[3 + i * 5](reference)
            reference = self.stream[4 + i * 5](reference)

        return reference,inverse

    def attention(self, inverse, discrimnative):
        """
        注意力模块的设置原理：
        1. 抽出反色流的第二层卷积结果，上采样到和判决流第一层卷积结果相同大小；
        2. 过一层conv提取特征
        3. 与判决流第一层卷积结果做内积再相加，类似skip connection
        4. 分出两路，一路过GAP过FC后再reszie到原大小，成为注意力
           注意力和另一路相乘
        """
        GAP = nn.AdaptiveAvgPool2d((1, 1))
        sigmoid = nn.Sigmoid()
        up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
        conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
        g = conv(up_sample)
        g = sigmoid(g)
        tmp = g * discrimnative + discrimnative
        f = GAP(tmp)
        f = f.view(f.size()[0], 1, f.size()[1])

        fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
        f = fc(f)
        f = sigmoid(f)
        f = f.view(-1, f.size()[2], 1, 1)
        out = tmp * f
        return out


class Net(nn.Module):
    def __init__(self,mod='thin'):
        super(Net, self).__init__()

        assert mod=='thin' or 'std',"model has only two variant: thin and std"
        if mod =='thin':
            self.stream = StreamThin()
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        else:
            self.stream = StreamThin()
            self.classifier = nn.Sequential(
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        self.GAP = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, inputs):
        half = inputs.size()[1] // 2
        reference = inputs[:, :half, :, :]
        reference_inverse = 255 - reference
        test = inputs[:, half:, :, :]
        del inputs
        test_inverse = 255 - test

        reference, reference_inverse = self.stream(reference, reference_inverse)
        test, test_inverse = self.stream(test, test_inverse)

        cat_1 = torch.cat((test, reference_inverse), dim=1)
        cat_2 = torch.cat((reference, test), dim=1)
        cat_3 = torch.cat((reference, test_inverse), dim=1)

        del reference, reference_inverse, test, test_inverse

        cat_1 = self.sub_forward(cat_1)
        cat_2 = self.sub_forward(cat_2)
        cat_3 = self.sub_forward(cat_3)

        return cat_1, cat_2, cat_3

    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, inputs.size()[1])
        out = self.classifier(out)

        return out



def early_stop(stop_round,loss,pre_loss,threshold=0.005):
    '''
    early stop setting
    :param stop_round: rounds under caculated
    :param pre_loss: loss list
    :param threshold: minimum one-order value of loss list
    :return: whether or not to jump out
    '''
    if(len(pre_loss)<stop_round):
        pre_loss.append(loss)
        return False
    else:
        loss_diff=np.diff(pre_loss,1)
        pre_loss.pop(0)
        pre_loss.append(loss)
        if(abs(loss_diff).mean()<threshold): # to low variance means flatten field
            return True
        else:
            return False