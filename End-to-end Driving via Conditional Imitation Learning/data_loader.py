# coding=utf-8

import os
import numpy as np
import h5py
import torch
import random
from torchvision import transforms
from torch.utils.data import Dataset

#使用开源的数据增强实现：https://github.com/aleju/imgaug 
import imgaug.augmenters as iaa
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
        #可直接使用的路径列表
        self.data_list = get_h5_list(data_path)
        self.transform = transform
        #数据集中每个文件为一个序列，其中包含200个状态动作对
        self.sequence_len = 200

    def __len__(self):
        #返回数据集大小
        return self.sequence_len*len(self.data_list)

    def __getitem__(self, idx):
        #提取对应下标的数据并将数据增强方法应用其中，此处下标是指状态动作对，因此一个文件中有200个
        #按照文件名顺序依次编号
        file_idx = idx // self.sequence_len
        sequence_idx = idx % self.sequence_len
        file_path_name = self.data_list[file_idx]
        
        with h5py.File(file_path_name,"r") as reader:
            #数据集包含两个datasets，具体可见https://github.com/carla-simulator/imitation-learning
            img = np.array(reader['rgb'])[sequence_idx]
            #应用图像增强
            img = self.transform(img)

            #读取高维控制信息，转向角度，油门，刹车，车速。
            target = np.array(reader['targets'])[sequence_idx]
            #统一数据类型
            target = np.array(target, dtype = np.float32)
            car_command = int(target[24])-2
            #除以90km/h是为了标准化
            car_speed = np.array([target[10]/90, ]).astype(np.float32)
            car_steer = target[0]
            car_gas = target[1]
            car_break = target[2]

            target_vec = np.zeros((4, 3), dtype=np.float32)
            target_vec[car_command,:] = [car_steer, car_gas, car_break]

            mask = np.zeros((4, 3), dtype=np.float32)
            mask[car_command,:] = 1

        return img, car_speed, target_vec.flatten(), mask.flatten()

#iaa中的数据增强方法需要位单张图片或者多张图片调用augment_image()或者augment_images()
#而transforms中的方法会将列表中的处理方法当作函数调用
#因此写一个类处理这种矛盾
class sometimes(object):
    def __init__(self, p, seq):
        self.p = p
        self.seq = seq

    def __call__(self, images):
        if self.p < random.random():
            return images
        return self.seq.augment_image(images)

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
            self.tran = transforms.Compose([
                        transforms.RandomOrder([
                            #各类数据增强方法,随机顺序，且按一定概率决定是不是使用
                            sometimes(0.09, iaa.GaussianBlur(sigma=(0,1.5))),
                            sometimes(0.09, iaa.AdditiveGaussianNoise(
                                                loc=0,
                                                scale=(0.0, 0.05),
                                                per_channel=0.5)),
                            sometimes(0.09, iaa.ContrastNormalization(
                                                (0.8, 1.2),
                                                per_channel=0.5)),
                            sometimes(0.3, iaa.Dropout(
                                                (0.0, 0.10),
                                                per_channel=0.5)),
                            sometimes(0.3, iaa.CoarseDropout(
                                                (0.0, 0.10),
                                                size_percent=(0.08, 0.2),
                                                per_channel=0.5)),
                            sometimes(0.3, iaa.Add(
                                                (-20, 20),
                                                per_channel=0.5)),
                            sometimes(0.4, iaa.Multiply(
                                                (0.9, 1.1),
                                                per_channel=0.2)),

                        ]),
                        transforms.ToTensor()
                    ])
        else:
            self.tran = transforms.Compose([
                    transforms.ToTensor()
                    ])

    def get_data_load(self):
        return torch.utils.data.DataLoader(
                    CARLA_Dataset(data_path=self.data_path, transform=self.tran),
                    batch_size=self.batch_size,
                    num_workers=4,
                    pin_memory=True,
                    shuffle=True
                )

