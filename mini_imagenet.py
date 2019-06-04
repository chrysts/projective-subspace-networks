import os.path as osp
from PIL import Image
import pickle as pkl
from torch.utils.data import Dataset
from torchvision import transforms
import os
import numpy as np

ROOT_PATH = './materials/'


class MiniImageNet(Dataset):



    def read_cache(self, data_path, split):
        cache_path = os.path.join(data_path,
                                  "mini-imagenet-cache-" + split + ".pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, "rb") as f:
                    data = pkl.load(f, encoding='bytes')
                    return data[b'image_data'], data[b'class_dict']
            except:
                with open(cache_path, "rb") as f:
                    data = pkl.load(f)
                    return data['image_data'], data['class_dict']


    def __init__(self, setname, data_path):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        data, dict = self.read_cache(data_path, setname)

        for l in lines:
            name, wnid = l.split(',')
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            label.append(lb)

        self.data = data
        self.label = label

        self.transform = transforms.Compose([
            transforms.ToTensor()#,
            #transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                     std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        img, label = self.data[i], self.label[i]
        #img = np.transpose(img, (2, 0, 1))
        image = self.transform(img)
        return image, label

