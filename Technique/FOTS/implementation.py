import torch
from data.dataset import ImageDataSet, collate_fn
from torch.utils.data import DataLoader
from config import opt

root_path = 'D:/_data/personal_project/ICDAR_2015/'
train_img = root_path + 'train_img'
train_txt = root_path + 'train_gt'

trainset = ImageDataSet(train_img, train_txt)
trainloader = DataLoader(
    trainset, batch_size=opt.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=opt.num_workers)

a, b, c, d = zip(*trainset)