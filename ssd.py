import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import  Variable
from math import  sqrt as sqrt
from itertools import  product as product
from data import coco, voc
from layers import *
import os

# 自定义SSD网络
class SSD(nn.Module):
    """
    SSD网络由VGG16进行改进，后增加multibox卷积层组成的

    phase: test/train
    size: 输入图片尺寸大小
    base: VGG16的层
    extras: 额外建立的层，用于把结果送至multibox的loc_layers和conf_layers中
    head: 包含一系列的loc和conf卷积层
    """
    # SSD模型初始化
    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile = True)
        self.size = size

        self.vgg = nn.ModuleList(base)
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim = -1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)  # 用于将预测结果转换成对应的坐标和类别编号形式, 方便可视化.

    # 定义forward函数，将设计好的layers和ops应用到输入图片上
    def forward(self, x):
        """
        Args:
            x, 输入的batch图片，Shape：[batch, 3, 300, 300]
        Return:
            Train:
                1: confidence layers, Shape: [batch*num_priors, num_classes] defalut box对应每个分类的置信度
                2: localization layers, Shape: [batch, num_priors*4] 每一个default box的四个坐标信息
                3: priorbox layers, Shape: [2, num_priors*4] 计算每个default box在同一尺度下的坐标
            Test:
                预测的类别标签，置信度，相关location
                Shape：[batch, topk, 7]
        """
        sources = list()# 存储参与预测的卷积层的输出(6个特征图)
        loc = list()# 用于存储预测的边框信息
        conf = list()# 用于存储预测的类别信息

        # 前向传播vgg至conv4_3 relu 得到第一个特征图
        for k in range(23):
            x = self.vgg[k](x)
        s = self.L2Norm(x) # 实现了L2归一化
        sources.append(s)

        # 继续向前传播vgg至FC7得到第二个特征图
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # 向前传播extras layers 得到其余四个特征图
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace = True)
            if k % 2 == 1:
                sources.append(x)

        # 应用multibox到source layers上, source layers中的元素均为各个用于预测的特征图谱
        # 将各个特征图中的定位和分类预测结果append进列表中
        for (x, l, c) in zip(sources, self.loc, self.conf):
            # permute重新排列维度顺序, PyTorch维度的默认排列顺序为 (N, C, H, W),
            # 因此, 这里的排列是将其改为 (N, H, W, C).
            # N:这批图像有几张 H:图像在竖直方向有多少像素 W:水平方向有多少像素 C:通道数
            # contiguous返回内存连续的tensor, 因为在执行permute或者transpose等操作之后, tensor的内存地址可能不是连续的
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())
        # cat 是 concatenate 的缩写, view返回一个新的tensor, 具有相同的数据但是不同的size, cat类似于numpy的reshape
        # 将除batch以外的其他维度合并, 因此, 对于边框坐标来说, 最终的shape为(两维):[batch, num_boxes*4]
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        # 相同的，最终的shape为(两维):[batch, num_boxes*num_classes]
        conf = torch.cat([o.view(o.size(0), -1)for o in conf], 1)

        # 训练阶段，直接返回结果求loss
        if self.phase == "train":
            result = (
                loc.view(loc.size(0), -1, 4), conf.view(conf.size(0), -1, self.num_classes), self.priors
            )

        # 测试阶段, 对定位和分类的预测结果进行分析得到最终的预测框
        # 使用detect对象，将预测出的结果进行解析，获得方便可视化的边框坐标和类别编号
        if self.phase == "test":
            result = self.detect(
                loc.view(loc.size(0), -1, 4), # 又将shape转换成: [batch, num_boxes, 4] -> [1, 8732, 4]
                self.softmax(conf.view(conf.size(0), -1, self.num_classes)), # [batch, num_boxes, num_classes] -> [1, 8732, 21]
                self.priors.type(type(x.data)) # 利用 PriorBox对象获取特征图谱上的 default box 该参数的shape为: [8732,4]，此方法与self.priors值相同
            )
        return result

    # 加载参数权重值
    def load_weights(self, base_file):
        oth, ext = os.path.splitext(base_file)
        if ext == ".pkl" or ".pth":
            print ("Loading the weights...")
            self.load_state_dict(torch.load(base_file, map_location = lambda storage, loc: storage))
            print("Finished!")
        else:
            print("Failed, only support '.pkl' or '.pth' files!")

sources = list()

# ssd采用改进后的vgg16网络结构，将原FC7层改为Conv7，并增加卷积深度继续添加Conv8_2, Conv9_2, Conv10_2, Conv11_2，最终结构变为conv1_2、conv2_2、conv3_3、conv4_3、conv5_3、conv6、conv7
# VGG16的两个全连接层被改掉（FC6,7),把全连接层换成1*1的卷积层的一个好处：输入图像的大小可以变化
def vgg(cfg, i, batch_norm = False):
    layers = []
    in_channels = i # 输入图像通道数，i = 3 因为输入是300*300*3的
    for v in cfg: # 循环建立多层，数据信息存放在一个字典中
        if v =='M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)] # ceil_mode为True则采用‘天花板模式’，把不足kernel_size的边保留，单独进行池化
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6) # dilation=卷积核元素之间的间距,扩大卷积感受野的范围，没有增加卷积size
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6, nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers

# SSD 模型中是利用多个不同层级上的 feature map 来进行同时进行边框回归和物体分类任务的
# 定义增加的额外层Conv8_2、Conv9_2、Conv10_2、Conv11_2用于feature scaling(用于执行回归和分类任务)，并采用步长为2的措施降低分辨率
def addextras(cfg, i, batch_norm = False):
    # extras8_1 = nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1)
    # extras8_2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
    # extras9_1 = nn.Conv2d(512, 128, 1, 1, 0)
    # extras9_2 = nn.Conv2d(128, 256, 3, 2, 1)
    # extras10_1 = nn.Conv2d(256, 128, 1, 1, 0)
    # extras10_2 = nn.Conv2d(128, 256, 3, 1, 0)
    # extras11_1 = nn.Conv2d(256, 128, 1, 1, 0)
    # extras11_2 = nn.Conv2d(128, 256, 3, 1, 0)
    # return [extras8_1, extras8_2, extras9_1, extras9_2, extras10_1, extras10_2, extras11_1, extras11_2]
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg): # S代表stride，为2时候就相当于缩小feature map
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = cfg[k+1], kernel_size = (1, 3)[flag], stride = 2, padding = 1)]
            else:
                layers += [nn.Conv2d(in_channels = in_channels, out_channels = v, kernel_size = (1, 3)[flag])]
            flag = not flag
        in_channels = v
    return layers

# 构建mutibox结构,创建6个特征图框的坐标点和分类类别
def multibox(vgg, extra_layers, cfg, num_classes):
    # ssd总会选择6个卷积特征图进行预测，分别为vgg的conv4_3, conv7 以及extras_layers的4段卷积的输出。
    # 因此，loc_layers 和 conf_layers 分别具有6个预测层.
    # loc_layers的输出维度是default box的种类(4or6)*4
    # conf_layers的输出维度是default box的种类(4or6)*num_class
    """
    选取的特征图在输入的两个list(vgg和add_extras)中的索引：
	vgg：21，-2
		conv4_3去掉relu的末端；conv7relu之前的1x1卷积；
	add_extras：1，3，5，7
		conv8_2末端；conv9_2末端；conv10_2末端；conv11_2末端
    """
    loc_layers = []
    conf_layers = []
    # 定义6个坐标预测层, 输出的通道数就是每个像素点上会产生的 default box 的数量
    loc1 = nn.Conv2d(vgg[21].out_channels, 4 * 4, 3, 1, 1)  # 利用conv4_3的特征图谱
    loc2 = nn.Conv2d(vgg[-2].out_channels, 6 * 4, 3, 1, 1)  # Conv7
    loc3 = nn.Conv2d(vgg[1].out_channels, 6 * 4, 3, 1, 1)  # exts8_2
    loc4 = nn.Conv2d(extras[3].out_channels, 6 * 4, 3, 1, 1)  # exts9_2
    loc5 = nn.Conv2d(extras[5].out_channels, 4 * 4, 3, 1, 1)  # exts10_2
    loc6 = nn.Conv2d(extras[7].out_channels, 4 * 4, 3, 1, 1)  # exts11_2
    loc_layers = [loc1, loc2, loc3, loc4, loc5, loc6]

    # 定义6个分类层, 对于每一个像素点上的每一个default box, 都需要预测出属于任意一个类的概率, 因此通道数为default box的数量乘以类别数.
    conf1 = nn.Conv2d(vgg[21].out_channels, 4 * num_classes, 3, 1, 1)
    conf2 = nn.Conv2d(vgg[-2].out_channels, 6 * num_classes, 3, 1, 1)
    conf3 = nn.Conv2d(extras[1].out_channels, 6 * num_classes, 3, 1, 1)
    conf4 = nn.Conv2d(extras[3].out_channels, 6 * num_classes, 3, 1, 1)
    conf5 = nn.Conv2d(extras[5].out_channels, 4 * num_classes, 3, 1, 1)
    conf6 = nn.Conv2d(extras[7].out_channels, 4 * num_classes, 3, 1, 1)
    conf_layers = [conf1, conf2, conf3, conf4, conf5, conf6]
    return loc_layers, conf_layers

#构建模型函数
def build_ssd(phase, size = 300, num_classes = 21):
    if phase != 'test' and phase != 'train':
        print("ERROR:The phase value" + phase + "is not available")
        return
    if size != 300:
        print("ERROR:Only 300 size is available. However, Your size is" + repr(size))
        return
    base_ssd, extras_ssd, head_ssd = multibox(vgg(base[str(size)], 3), addextras(extras[str(size)], 1024), mbox[str(size)], num_classes)
    return SSD(phase, size, base_ssd, extras_ssd, head_ssd, num_classes)

#vgg网络结构参数
base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256,256, 'C', 512, 512, 512, 'M', 512, 512, 512], #各卷积层通道数
    '512': [],
}
#extras层参数
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
#multibox相关参数
mbox = {
    '300': [4, 6, 6, 6, 4, 4],
    '500': []
}

