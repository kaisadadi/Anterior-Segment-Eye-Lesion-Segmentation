import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class BCEWithLogitsLoss2d(nn.Module):

    def __init__(self, weight, size_average=True):
        super(BCEWithLogitsLoss2d, self).__init__()
        self.size_average = size_average
        assert len(weight) == 12
        self.weight = weight  #list

    def forward(self, predict, target):
        """
            Args:
                predict:(n, 1, h, w)
                target:(n, 1, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 4
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(2), "{0} vs {1} ".format(predict.size(2), target.size(2))
        assert predict.size(3) == target.size(3), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        loss = 0
        for idx in range(predict.shape[1]):
            label_idx = target[:, idx, :, :]
            pred_idx = torch.clamp(predict[:, idx, :, :], 1e-6, 1 - 1e-6)
            loss_idx =  - self.weight[idx] * label_idx * torch.log(pred_idx) - (1 - label_idx) * torch.log(1 - pred_idx)
            loss += torch.mean(loss_idx)
        return loss / 12


class CrossEntropy2d(nn.Module):

    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return Variable(torch.zeros(1))
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config):
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = config.getint("model", "output_dim")
        self.criterion = []
        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
                # print_info("Task %d with weight %.3lf" % (task, ratio))
            except Exception as e:
                self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss


def multi_label_cross_entropy_loss(outputs, labels, length=None):
    labels = labels.float()
    labels = torch.clamp(labels, 0, 1)
    temp = torch.clamp(outputs, 1e-6, 1-1e-6)
    bs = outputs.shape[0]
    loss = 0
    for idx in range(bs):
        res = - labels[idx] * torch.log(temp[idx]) - (1 - labels[idx]) * torch.log(1 - temp[idx])
        #for k in range(res.shape[0]):
        #    if any(torch.isnan(res[k])):
        #        print(res[k])
        #        gg
        if length != None:
            res = torch.mean(torch.sum(res[:length[idx]], dim=1))
        loss += res
    return loss / bs


def cross_entropy_loss(outputs, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)
