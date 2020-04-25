# coding=utf-8

import os
import numpy as np
import h5py
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from imgaug import augmenters as iaa
from toolkit import get_h5_list

#数据读取主要使用torch.utils.data.DataLoader，其需要Dataset类型
#torch.utils.data.Dataset是抽象类，通过继承Dataset类并重写__len__与__getitem__方法实现自定义数据读取

#继承Dataset，重写关键函数。为了避免中文显示问题，函数注解用英文。
class CARLA_Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        Args:
            data_path (string): Path to the data folder.    
            transform (): transform to be applied on a sample.
        """
        self.data_path = data_path
        self.data_list = get_h5_list(data_path)

    def __len__(self):
        #返回数据集大小，待实现
        return len()

    def __getitem__(self, idx):
        #提取对应下表的数据并将数据增强方法应用其中，待实现
        return sample

#为了方便处理训练与测试是需要的不同数据增强方法，再封一层
class CARLA_Data():
    #使用时实例化此类，之后使用get_data_load()函数获取对应的dataloader
    def __init__(self, data_path, train_eval_flag, batch_size):
        """
        Args:
            data_path (string): Path to the data folder.    
            train_eval_flag (string): "train" or "eval".
            batch_size (int): Recommended to be consistent with the number of processor cores
        """
        
        self.data_path = data_path
        self.batch_size = batch_size


        if train_eval_flag == "train":
            transforms = transforms.Compose([
                        transforms.RandomOrder([
                            #各类数据增强方法,待实现


                        ])
                        transforms.ToTensor()
                    ])

    def get_data_load():
        return torch.utils.data.DataLoader(
                    CARLA_Dataset(data_path=self.data_path, transform=transforms),
                    batch_size=self.batch_size,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=True
                )

