import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np
import pandas as pd
import pickle
import os


def deap_preprocess(data_file, dimension, dataset_dir):
    """
    加载和预处理DEAP数据

    Args:
        data_file: 被试文件名 (e.g., 's01')
        dimension: 'arousal' 或 'valence'
        dataset_dir: 数据集根目录
    """
    rnn_suffix = ".mat_win_384_rnn_dataset.pkl"
    label_suffix = ".mat_win_384_labels.pkl"
    with_or_without = 'yes'

    full_dataset_dir = os.path.join(dataset_dir, "384",
                                    f"{with_or_without}_{dimension}")

    # 加载数据
    with open(os.path.join(full_dataset_dir, data_file + rnn_suffix), "rb") as fp:
        rnn_datasets = pickle.load(fp)
    with open(os.path.join(full_dataset_dir, data_file + label_suffix), "rb") as fp:
        labels = pickle.load(fp)
        labels = np.transpose(labels)

    # One-hot编码
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)

    # 打乱数据
    index = np.array(range(0, len(labels)))
    np.random.shuffle(index)
    rnn_datasets = rnn_datasets[index]
    labels = labels[index]

    # 重塑数据 (samples, time, channels, 1)
    datasets = rnn_datasets.reshape(-1, 384, 32, 1).astype('float32')
    labels = labels.astype('float32')

    return datasets, labels

class DEAPDataset(Dataset):
    """DEAP数据集类"""

    def __init__(self, data, labels):
        self.data = torch.from_numpy(data).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 原始数据形状: (time=384, channels=32, 1) from deap_preprocess
        # 需要输出: (channels=32, time=384)
        sample = self.data[idx]  # (384, 32, 1)

        # 移除最后一个维度
        if sample.dim() == 3 and sample.shape[2] == 1:
            sample = sample.squeeze(2)  # (384, 32)

        # 转置: (384, 32) -> (32, 384)
        sample = sample.transpose(0, 1)  # (32, 384)

        label = torch.argmax(self.labels[idx])  # 转换为类别索引
        return sample, label