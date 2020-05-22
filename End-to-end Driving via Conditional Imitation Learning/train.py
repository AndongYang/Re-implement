# coding = utf-8

import argparse
import time
import math
import shutil
import datetime
import logging
import os

import torch
import torch.nn as nn

from net import Net
from data_loader import CARLA_Data

parser = argparse.ArgumentParser(description='training arg')
parser.add_argument('--gpu', default='1', type=str, help='GPU id to use.')
parser.add_argument('--lr', '--learning-rate', default=2e-4, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--batch_size', default=32, type=int, metavar='BS',
                    help='batch size of training')
#这两个目录是含有h5文件的文件夹
parser.add_argument('--train_data_dir', default="./train_data/",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--eval_data_dir', default="./eval_data/",
                    type=str, metavar='PATH',
                    help='training dataset')
parser.add_argument('--epoch_num', default=2, type=int, metavar='N',
                    help='epoch number')
parser.add_argument('--speed_weight', default=1, type=float,
                    help='speed weight')
parser.add_argument('--branch_weight', default=1, type=float,
                    help='branch weight')
parser.add_argument('--models_save', default='./models_save/checkpoint.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--print_freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')

def output_log(data, logger=None):
    print("{}:{}".format(datetime.datetime.now(), data))
    if logger is not None:
        logger.critical("{}:{}".format(datetime.datetime.now(), data))


start_time = time.time()
def main():
    #获得参数，初始化日志
    global args
    args = parser.parse_args()
    logging.basicConfig(filename="./training.log",
                    level=logging.ERROR)

    #获取训练数据
    data = CARLA_Data(
        data_path = args.train_data_dir,
        train_eval_flag = 'train',
        batch_size = args.batch_size
    )
    data_loader = data.get_data_load()
    #获取测试数据
    eval_data = CARLA_Data(
        data_path = args.eval_data_dir,
        train_eval_flag = 'eval',
        batch_size = args.batch_size
    )
    eval_data_loader = eval_data.get_data_load()

    #指定使用的显卡
    GLOBAL_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICE']=args.gpu
    
    #实例化网络
    model = Net().to(GLOBAL_DEVICE)

    #定义loss，优化器
    loss_cal = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), args.lr, betas=(0.7, 0.85))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    #恢复checkpoint
    os.makedirs('./models_save/', exist_ok=True)
    if os.path.isfile(args.models_save):
        output_log("=> loading checkpoint '{}'".format(args.models_save),
                    logging)
        checkpoint = torch.load(args.models_save)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['scheduler'])
        output_log("=> loaded checkpoint '{}' "
                    .format(args.models_save), logging)
    else:
        output_log("=> no checkpoint found at '{}'".format(args.models_save),logging)

    #记录最准确的模型，方便最后输出参数
    best_prec = math.inf
    
    for epoch in range(args.epoch_num):
        do_train(data_loader, model, loss_cal, optimizer, epoch)
        lr_scheduler.step()

        #添加一个eval函数，测试效果怎么样，然后保存效果最好的参数
        prec = evaluation(eval_data_loader, model, loss_cal, epoch)

        best_prec = min(prec, best_prec)
        torch.save({'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'scheduler': lr_scheduler.state_dict(),
            'optimizer': optimizer.state_dict()}, args.models_save)
        if prec <= best_prec:
            shutil.copyfile(
            args.models_save,
            os.path.join("./models_save/", "checkpoint_best.pth")
            )


def do_train(loader, model, loss_cal, optimizer, epoch):
    model.train()

    for i, (img, car_speed, target, mask) in enumerate(loader):
        #branch_info = [car_steer, car_gas, car_break]

        #如果GPU可用，将数据移入指定硬件
        if torch.cuda.is_available():
            img = img.cuda(non_blocking=True)
            car_speed = car_speed.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            mask = mask.cuda(non_blocking=True)
        
        #0 Follow lane, 1 Left, 2 Right, 3 Straight
        #branches_out 4*3
        branches_out, pred_speed = model(img, car_speed)

        #将branches_out除了本次命令之外的所有值置为零
        #element-wise操作
        mask_out = branches_out*mask
        branches_loss = loss_cal(mask_out,target) * 4 
        speed_loss = loss_cal(pred_speed, car_speed)

        #计算整体loss
        loss = args.branch_weight * branches_loss + \
                args.speed_weight * speed_loss
        
        #反传优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % args.print_freq == 0 or i == len(loader):
            output_log(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time:.3f}\t'
                'Branch loss {branch_loss:.3f}\t'
                'Speed loss {speed_loss:.3f}\t'
                'Loss {loss:.4f}\t'
                .format(
                    epoch, i, len(loader), batch_time=time.time() - start_time,
                    branch_loss=branches_loss, speed_loss=speed_loss, 
                    loss=loss), logging)


#每隔一段进行一次测试，记录平均损失值，方便选出效果最好的模型
def evaluation(loader, model, loss_cal, epoch):
    model.eval()

    avg_loss = 0
    count = 0
    #不反传
    with torch.no_grad():
        for i, (img, car_speed, target, mask) in enumerate(loader):
            #如果GPU可用，将数据移入指定硬件
            if torch.cuda.is_available():
                img = img.cuda(non_blocking=True)
                car_speed = car_speed.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
                mask = mask.cuda(non_blocking=True)

            #0 Follow lane, 1 Left, 2 Right, 3 Straight
            #branches_out 4*3
            branches_out, pred_speed = model(img, car_speed)

            #将branches_out除了本次命令之外的所有值置为零
            #element-wise操作
            mask_out = branches_out*mask
            branches_loss = loss_cal(mask_out,target) * 4 
            speed_loss = loss_cal(pred_speed, car_speed)

            #计算整体loss
            loss = args.branch_weight * branches_loss + \
                    args.speed_weight * speed_loss

            avg_loss+=loss
            count+=1

    return 1.0*avg_loss/count

if __name__ == '__main__': 
    main()
