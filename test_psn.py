import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import ConvNet
from algorithm.subspace_projection import Projection
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/psn-model/max-acc.pth')
    parser.add_argument('--batch', type=int, default=600)
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--data-path', default='/home/csimon/research/data/miniimagenet/split/') # need to change to your data path
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test', args.data_path)
    sampler = CategoriesSampler(dataset.label,
                                args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=8, pin_memory=True)

    model = ConvNet().cuda()
    model.load_state_dict(torch.load(args.load))
    model.eval()
    projection_pro = Projection(shot=2) #subspace dim

    ave_acc = Averager()
    acc_all=[]

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        #data, _ = [_ for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        proto = model(data_shot)
        proto = proto.reshape(args.shot, args.way, -1)
        proto = torch.transpose(proto, 0, 1)
        hyperplanes,  mu = projection_pro.hyperplanes(proto, args.way, args.shot)

        logits, _ = projection_pro.projection_metric(model(data_query), hyperplanes, mu=mu)

        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)
        #label = label.type(torch.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        acc_all.append(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))

    std = np.std(acc_all) * 1.96 / np.sqrt(args.batch)
    print('Final {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, std * 100))