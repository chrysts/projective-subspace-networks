import os
import shutil
import time
import pprint

import torch
import torch.nn as nn
import torch.autograd.variable as Variable



class GaussianNoise(nn.Module):

    def __init__(self, batch_size, input_shape=(3, 84, 84), std=0.05):
        super(GaussianNoise, self).__init__()
        self.shape = (batch_size,) + input_shape
        self.noise = Variable(torch.zeros(self.shape).cuda())
        self.std = std

    def forward(self, x, std=0.15):
        noise = Variable(torch.zeros(x.shape).cuda())
        noise = noise.data.normal_(0, std=std)
        return x + noise


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.mkdir(path)
    else:
        os.mkdir(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    #logits = -((a - b)**2).sum(dim=2)
    logits = -((a - b)**2).sum(dim=2)
    return logits


class Timer():

    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

def set_protocol(data_path, protocol, test_protocol):
    train = []
    val = []

    all_set = ['shn', 'hon', 'clv', 'clk', 'gls', 'scl', 'sci', 'nat', 'shx', 'rel']

    if protocol == 'p1':
        for i in range(3):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p2':
        for i in range(3, 6):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p3':
        for i in range(6, 8):
            train.append(data_path + '/crops_' + all_set[i])
    elif protocol == 'p4':
        for i in range(8, 10):
            train.append(data_path + '/crops_' + all_set[i])

    if test_protocol == 'p1':
        for i in range(3):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p2':
        for i in range(3, 6):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p3':
        for i in range(6, 8):
            val.append(data_path + '/crops_' + all_set[i])
    elif test_protocol == 'p4':
        for i in range(8, 10):
            val.append(data_path + '/crops_' + all_set[i])



    return train, val