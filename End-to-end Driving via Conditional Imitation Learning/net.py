# coding=utf-8


import torch
import torch.nn as nn


class Net(nn.Module):
    #论文使用参数：self.dropout_vec = [1.0] * 8 + [0.7] * 2 + [0.5] * 2 + [0.5] * 1 + [0.5, 1.] * 5
    def __init__(self, dropout_vec=None):
        super(CarlaNet, self).__init__()
        self.conv_block = nn.Sequential(
            #200*88->98*42
            nn.Conv2d(3, 32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            #98*42->96*40
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            #96*40->47*19
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            #47*19->45*17
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            #45*17->22*8
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #22*8->20*6
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            #20*6->18*4
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            #18*4->16*2
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.img_out_fc = nn.Sequential(
                nn.Linear(16*2*256, 512),
                nn.Dropout(0.3),
                nn.ReLU(),

                nn.Linear(512, 512),
                nn.Dropout(0.3),
                nn.ReLU(),
            )

        self.speed_in = nn.Sequential(
                nn.Linear(1, 128),
                nn.Dropout(0.5),
                nn.ReLU(),

                nn.Linear(128, 128),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.img_speed_fc = nn.Sequential(
                nn.Linear(512+128, 512),
                nn.Dropout(0.5),
                nn.ReLU(),
            )

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),

                nn.Linear(256, 256),
                nn.ReLU(),

                nn.Linear(256, 3),
            ) for i in range(4)
        ])

        self.speed_out = nn.Sequential(
                nn.Linear(512, 256),
                nn.Dropout(0.5),
                nn.ReLU(),

                nn.Linear(256, 256),
                nn.ReLU(),

                nn.Linear(256, 1),
            )

    def forward(self, img, speed):
        #图像卷积与展平
        img = self.conv_block(img)
        img = img.view(-1, 8192)
        img = self.img_out_fc(img)

        #速度输入
        speed = self.speed_in(speed)

        #速度与图像拼接，此时图像为batch*512，速度为batch*128
        img_speed = torch.cat([img, speed], dim=1)
        img_speed = self.img_speed_fc(img_speed)

        #为所有4个分支都计算输出
        #0 Follow lane, 1 Left, 2 Right, 3 Straight
        #注意数据集给出的值是从2开始，需要减2才从0开始
        branches_out = torch.cat([out(img_speed) for out in self.branches], dim=1)

        #计算预测速度
        pred_speed = self.speed_out(img)

        return branches_out, pred_speed
