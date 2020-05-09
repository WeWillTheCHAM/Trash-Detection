import argparse
import torch
import sys
import numpy as np
import os
import time
import visdom
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd

def str2bool(v):
    return v.lower() in ("yes", "true", "t", 1)

# 引进parser的好处就是linux系统方便操作，不需要在里面修改数据集信息，在外面通过语句就行，window系统要进入代码里修改信息
parser = argparse.ArgumentParser(description = "Single Shot MultiBox Detection")
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO'],
                    type=str, help='VOC or COCO')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
# 预训练模型
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
# 批处理数
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
# 迭代起始点
parser.add_argument('--start_iter', default=0, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
# learning rate 学习率
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
# 使用visdom进行损失可视化
parser.add_argument('--visdom', default=False, type=str2bool,
                    help='Use visdom for loss visualization')
# 模型保存地址
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
    else:
        torch.set_default_tensor_type("torch.FloatTensor")
else:
    torch.set_default_tensor_type("torch.FloatTensor")

def train():
    if args.dataset == "COCO":
        if args.dataset_root == VOC_ROOT:
            if not os.path.exists(COCO_ROOT):
                parser.error('Must specify dataset_root if specifying dataset')
            print("WARNING: Using default COCO dataset_root because " +
                  "--dataset_root was not specified.")
            args.dataset_root = COCO_ROOT
        cfg = coco # coco位于config.py文件中
        # COCODetection类 位于coco.py文件中
        # SSDAugmentation类 位于utils/augmentations.py文件中
        dataset = COCODetection(root = args.dataset_root, transform = SSDAugmentation(cfg["min_dim"], MEANS))
    elif args.dataset == "VOC":
        if args.dataset_root == COCO_ROOT:
            parser.error('Must specify dataset if specifying dataset_root')
        cfg = voc
        dataset = VOCDetection(root = args.dataset_root, transform = SSDAugmentation(cfg["min_dim"], MEANS))

    if args.visdom:
        import visdom
        viz = visdom.Visdom()
    ssd_net = build_ssd("train", cfg["min_dim"], cfg["num_classes"])
    net = ssd_net

    if args.cuda:
        net = torch.nn.DataParallel(ssd_net)
        cudnn.benchmark = True # 这个flag可以让内置的cuDNN的auto-tuner自动寻找最适合当前配置的算法.

    # resume 类型为 str, 值为checkpoint state_dict file
    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.save_folder + args.basenet)
        print('Loading base network...')
        ssd_net.load_state_dict(vgg_weights)

    # 将所有的参数都移送到GPU内存中
    if args.cuda:
        net = net.cuda()

    # 用xavier方法初始化新添加层的权重
    if not args.resume:
        print('Initializing weights...')
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    # 随机梯度下降优化，计算梯度和误差并更新参数
    # SGD在学习中增加了噪声，有正则化的效果
    # 学习效率进行线性衰减可以保证SGD收敛
    optimizer = optim.SGD(net.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    # MultiBoxLoss类 位于layers/modules/multibox_loss.py文件中
    criterion = MultiBoxLoss(cfg["num_classes"], 0.5, True, 0, True, 3, 0.5, False, args.cuda)

    net.train()
    # Loss计数器
    loc_loss = 0
    conf_loss = 0
    epoch = 0
    print('Loading the dataset...')

    epoch_size = len(dataset)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    if args.visdom:
        vis_title ='SSD.PyTorch on ' +dataset.name
        vis_legend = ['Loc loss', 'Conf loss', 'Total loss']
        iter_plot = create_vis_plot('Iteration', 'Loss', vis_title, vis_legend)
        epoch_plot = create_vis_plot('Epoch', 'Loss', vis_title, vis_legend)

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers = args.num_workers, shuffle = True, collate_fn = detection_collate, pin_memory = True)

    # 创建batch迭代器
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg["max_iter"]):
        if args.visdom and iteration != 0 and (iteration % epoch_size == 0):
            update_vis_plot(epoch, loc_loss, conf_loss, epoch_plot, None, "append", epoch_size)
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        if iteration in cfg["lr_steps"]:
            step_index += 1
            adjust_learning_rate(optimizer, args.gamma, step_index)

        # 加载训练数据
        images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile = True) for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile = True) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)

        # backprop
        optimizer.zero_grad()
        loss_loc, loss_conf = criterion(out, targets)
        loss = loss_loc + loss_conf
        loss.backward()
        optimizer.step()
        t1 = time.time()
        loc_loss += loss_loc.data[0]
        conf_loss += loss_conf.data[0]

        # 每隔10次迭代就输出一次训练状态信息
        if iteration % 10 == 0:
            print('timer: %.4f sec.' % (t1 - t0))
            print('iter ' + repr(iteration) + ' ||Loss: %.4f ||' % (loss.data[0]), end = ' ')

        if args.visdom:
            update_vis_plot(iteration, loss_loc.data[0], loss_conf.data[0], iter_plot, epoch_plot, 'append')

        # 迭代多少次保存一个模型以及保存的文件名
        if iteration != 0 and iteration % 5000 == 0:
            print("Saving state, iter: ", iteration)
            torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' + repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + '' + args.dataset + '.pth')

# 调整learning rate 每经过固定迭代次数, 就将lr衰减1/10
def adjust_learning_rate(optimizer, gamma, step):
    lr = args.lr * (gamma ** (step))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 创建可视化的线图
def create_vis_plot(x_label, y_label, _title, _legend):
    return viz.line(
        x = torch.zeros((1,)).cpu(),
        y = torch.zeros((1, 3)).cpu(),
        opts = dict(
            xlabel = x_label,
            ylabel = y_label,
            title = _title,
            legend = _legend
        )
    )

# 更新可视化的线图
def update_vis_plot(iteration, loc, conf, window1, window2, update_type,epoch_size = 1):
    viz.line(
        X = torch.ones((1, 3)).cpu() * iteration,
        Y = torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu() / epoch_size,
        win = window1,
        update = update_type
    )
    if iteration == 0:
        viz.line(
            X = torch.zeros((1, 3)).cpu(),
            Y = torch.Tensor([loc, conf, loc + conf]).unsqueeze(0).cpu(),
            win = window2,
            update = True
        )

def xavier(param):
    init.xavier_uniform(param)

# 对网络参数执行Xavier初始化
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()

if __name__ == '__main__':
    train()