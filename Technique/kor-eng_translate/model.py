import pandas as pd, os
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim

# from torchtext.datasets import TranslationDataset, Multi30k
# from torchtext.data import Field #, BucketIterator
# from torchtext.legacy.datasets import TranslationDataset, Multi30k
# from torchtext.legacy.data import Field, BucketIterator

import spacy
import numpy as np

import random
import math
import time

data = glob('./한국어-영어 번역(병렬) 말뭉치/*.xlsx')

df = pd.DataFrame(columns = ['원문','번역문'])
path = './한국어-영어 번역(병렬) 말뭉치/'

file_list = os.listdir(path)

for data in tqdm(file_list):
    temp = pd.read_excel(path+data)
    df = pd.concat([df,temp[['원문','번역문']]])

