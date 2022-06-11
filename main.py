# -*- coding: utf-8 -*-
# ---
# @File: main.py
# @Author: sgdy3
# @E-mail: sgdy03@163.com
# @Time: 2022/6/5
# Describe: IDN的实现
# source: https://github.com/wk-ff/IDN
# ---

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from torch import optim
import time
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from model import Net
from DataConfig import dataset


def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]

    # majority vote
    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.bce_loss = nn.BCELoss()


    def forward(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)
        # print(max(x), max(label))
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)


def eval_stage(model,data_loader):
    """
    用模型预测数据集上的结果，画图并返回预测值
    """
    result=[]
    label=[]
    with torch.no_grad():
        it=iter(data_loader)
        for i in range(len(data_loader)):
            inputs,labels=next(it)
            if cuda:
                inputs,labels=inputs.cuda(),labels.cuda()
            pred=model(inputs)
            result.append((0.3*pred[0]+0.4*pred[1]+0.3*pred[2]).cpu().numpy())
            label.append(labels.cpu())
            print(f'finish {i}')
    result=np.vstack(result)
    label=np.hstack(label)
    draw_fig(result,label)
    return result,label


def draw_fig(pred,label):
    fpr, tpr, thresholds = roc_curve(label,pred, pos_label=1)
    fnr = 1 -tpr
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))] # We get EER when fnr=fpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))] # judging threshold at EER
    pred_label=pred.copy()
    pred_label[pred_label>eer_threshold]=1
    pred_label[pred_label<=eer_threshold]=0
    acc=(pred_label==label).sum()/label.size
    pred_label=pred.copy()
    pred_label[pred_label>0.5]=1
    pred_label[pred_label<=0.5]=0
    acc_half=(pred_label==label).sum()/label.size

    area = auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.5f)' % area)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC on testing set')
    plt.legend(loc="lower right")
    plt.show()

    return area,EER,acc,acc_half

if __name__ == '__main__':
    model=Net()

    BATCH_SIZE = 32
    EPOCHS = 1
    LEARNING_RATE = 0.001
    save=False

    np.random.seed(0)
    torch.manual_seed(4)

    cuda = torch.cuda.is_available()

    train_set = dataset(train=True)
    test_set = dataset(train=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    # 检验模型前向传播是否正常
    # test,lab=next(iter(train_loader))
    # test=test.cuda()
    # lab=lab.float()
    # lab=lab.cuda()

    if cuda:
        model = model.cuda()
    criterion = loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # 创建日志
    writer = SummaryWriter(log_dir='./logs/scalar')

    if cuda:
        criterion = criterion.cuda()
    iter_n = 0

    t = time.strftime("%m-%d-%H-%M", time.localtime())
    print(len(train_loader))
    for epoch in range(1, EPOCHS + 1):
        print(f'epoch{epoch} start')
        for i, (inputs, labels) in enumerate(train_loader):
            torch.cuda.empty_cache()

            optimizer.zero_grad()

            labels = labels.float()
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()

            predicted = model(inputs)

            loss = criterion(*predicted, labels)

            loss.backward()
            optimizer.step()

            accuracy = compute_accuracy(predicted, labels)

            # writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
            # writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)

            # 每100epochs在测试集上评估结果
            # if i % 100== 0:
            #     with torch.no_grad():
            #         accuracys = []
            #         for i_, (inputs_, labels_) in enumerate(test_loader):
            #             labels_ = labels_.float()
            #             if cuda:
            #                 inputs_, labels_ = inputs_.cuda(), labels_.cuda()
            #             predicted_ = model(inputs_)
            #             accuracys.append(compute_accuracy(predicted_, labels_))
            #         accuracy_ = sum(accuracys) / len(accuracys)
            #         writer.add_scalar(t+'/test_accuracy', accuracy_, iter_n)
            #     print('test loss:{:.6f}'.format(accuracy_))

            iter_n += 1

            if i == 200 and save:
                torch.save(model.state_dict(), './NetWeights/IDN/IDN.pth')
                break

            print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch, EPOCHS, i, loss.item(), accuracy))
    writer.close()

    result,label=eval_stage(model,test_loader)