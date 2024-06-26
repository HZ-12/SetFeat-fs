import threading

from einops import rearrange, repeat
from torch import nn
import numpy as np
import torch
import math
# from models.utils import *
# from dataloader.data_utils import *
from torch.autograd import Variable
import torch.nn.functional as F
import pprint
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import random
import os

from tqdm import tqdm

from dataloader.ml_dataFunctions import SetDataManager


def save_list_to_txt(name, input_list):
    f = open(name, mode='w')
    for item in input_list:
        f.write(item + '\n')
    f.close()


def fun_metaLoader(args, normalization, n_eposide=400, file_name='/val.json'):
    val_file = args.data_dir + args.dataset + file_name
    val_datamgr = SetDataManager(args, normalization, args.way, args.shot, args.query, n_eposide)
    return val_datamgr.get_data_loader(val_file)


def set_seed(seed):
    if seed == 0:
        torch.backends.cudnn.benchmark = True
    else:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def ensure_path(path):
    if os.path.exists(path):
        pass
    else:
        print('create folder:', path)
        os.makedirs(path)


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
    if torch.cuda.is_available():
        return (pred == label).type(torch.cuda.FloatTensor).mean().item()
    else:
        return (pred == label).type(torch.FloatTensor).mean().item()


_utils_pp = pprint.PrettyPrinter()


def pprint(x):
    _utils_pp.pprint(x)


def compute_confidence_interval(data):
    """
    Compute 95% confidence interval
    :param data: An array of mean accuracy (or mAP) across a number of sampled episodes.
    :return: the 95% confidence interval for this data.
    """
    a = 1.0 * np.array(data)
    m = np.mean(a)
    std = np.std(a)
    pm = 1.96 * (std / np.sqrt(len(a)))
    return m, pm


def dataextractor(batch, ys, yq, num_class, shot):
    if torch.cuda.is_available():
        data, _ = [_.cuda() for _ in batch]
    else:
        data, _ = batch
    xs_, xq_ = data[:num_class * shot], data[num_class * shot:]
    for w in range(num_class):
        if w == 0:
            xs = xs_[ys == w]
            xq = xq_[yq == w]
        else:
            xs = torch.cat((xs, xs_[ys == w]), dim=0)
            xq = torch.cat((xq, xq_[yq == w]), dim=0)
    return torch.cat((xs, xq), dim=0)


g_means = []
g_std = []


def calculateMeanAndStd(scope):
    # 挑选多少图片进行计算
    img_h, img_w = 84, 84
    imgs = np.zeros([img_w, img_h, 3, 1])
    means, std = [], []
    # shuffle , 随机挑选图片
    for i in tqdm(scope):
        # img_path = os.path.join('F:\\sshCode\\FewTURE\\datasets\\dataloaders\\ocean\\images',
        #                         lines[i].split(',')[0])
        img_path = os.path.join('benchmarks/ocean/images',
                                lines[i].split(',')[0].replace("\\", "/"))
        img = cv2.imread(img_path)
        img = cv2.resize(img, (img_h, img_w))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)
        #  print(i) imgs = imgs.astype(np.float32)/255.
    for i in tqdm(range(3)):
        pixels = imgs[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        std.append(np.std(pixels))
        # cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转means.reverse()
        # BGR --> RGBstdevs.reverse()
    print("normMean = {}".format(means))
    print("normStd = {}".format(std))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, std))
    g_means.append(means)
    g_std.append(std)


# train_csv_path = "F:\\sshCode\\FewTURE\\datasets/dataloaders/ocean/split/train.csv"
# for server
train_csv_path = "benchmarks/ocean/split/train.csv"
import cv2
import random


CNum = 10000
with open(train_csv_path, 'r') as f:
    lines = f.readlines()[1:]
    random.shuffle(lines)
    thread1 = threading.Thread(target=calculateMeanAndStd, args=(range(1000)))
    thread2 = threading.Thread(target=calculateMeanAndStd, args=(range(1000, 2000)))
    thread3 = threading.Thread(target=calculateMeanAndStd, args=(range(2000, 3000)))
    thread4 = threading.Thread(target=calculateMeanAndStd, args=(range(3000, 4000)))
    thread5 = threading.Thread(target=calculateMeanAndStd, args=(range(4000, 5000)))
    thread6 = threading.Thread(target=calculateMeanAndStd, args=(range(5000, 6000)))
    thread7 = threading.Thread(target=calculateMeanAndStd, args=(range(6000, 7000)))
    thread8 = threading.Thread(target=calculateMeanAndStd, args=(range(7000, 8000)))
    thread9 = threading.Thread(target=calculateMeanAndStd, args=(range(8000, 9000)))
    thread10 = threading.Thread(target=calculateMeanAndStd, args=(range(9000, 10000)))

    thread1.start()
    thread2.start()
    thread3.start()
    thread4.start()
    thread5.start()
    thread6.start()
    thread7.start()
    thread8.start()
    thread9.start()
    thread10.start()

    thread1.join()
    thread2.join()
    thread3.join()
    thread4.join()
    thread5.join()
    thread6.join()
    thread7.join()
    thread8.join()
    thread9.join()
    thread10.join()

    print("normMean = {}".format(np.mean(g_means)))
    print("normStd = {}".format(np.mean(g_std)))
    print('transforms.Normalize(normMean = {}, normStd = {})'.format(np.mean(g_means), np.mean(g_std)))


