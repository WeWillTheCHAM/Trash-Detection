import sys
import os
import torch
import torch.utils.data as data
import cv2
import numpy as np
if sys.version_info[0] == 2: #sys.version_info(major=3, minor=6, micro=2, releaselevel='final', serial=0)
    import xml.etree.cElementTree as ET   #解析xml文件
else:
    import xml.etree.ElementTree as ET

# 识别的物体类别
VOC_CLASSES = (
    'DisposableFastFoodBox', 'StainedPlastic', 'CigaretteButts', 'ToothPick', 'Dishes',
    'BambooChopsticks', 'Peel', 'Eggshell', 'PowerBank', 'Backpack', 'CosmeticBottle', 'PlasticToys',
    'PlasticTableware', 'PlasticHangers', 'OldClothers', 'Cans', 'Pillow', 'StuffedToy', 'GlassBottle',
    'LeatherShoes', 'ChoppingBoard', 'CardboardBox', 'WineBottles', 'MetalFoodCan', 'Pot', 'DryCell',
    'Ointment', 'ExpiredDrugs'
)

# data数据读取处
VOC_ROOT = os.path.join(os.getcwd(), "/WasteDataset/")


class VOCAnnotationTransform(object):
    '''
    将VOC annotations转换为bbox坐标的张量和标签索引，并对索引的类名的字典查找进行初始化
    解析xml文档

    Arguments:
        class_to_ind(dict, optional):dictionary lookup of classnames -> indexes
        keep_difficult(bool, optional): keep difficult instances or not (default:False)
        height(int): height
        width(int)：width
    '''

    def __init__(self, class_to_ind=None, keep_difficult=False):
        self.class_to_ind = class_to_ind or dict(zip(VOC_CLASSES, range(len(VOC_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target, width, height):
        """
        :param target(annotation): the target annotation to be made usable will be an ET.Element
        :param width:
        :param height:
        :return:
            a list containing lists of bounding box [bbox coords, class name]
        """
        res = []
        for obj in target.iter('object'):  #寻找object元素，即xml中标注的对象
            difficult = int(obj.find('difficult').text) == 1     #寻找difficult子元素
            if not self.keep_difficult and difficult:  #如果是difficult则跳过该图片
                continue

            name = obj.find('name').text.lower().strip() #将类别名称提取出来
            bbox = obj.find('bndbox')  #框坐标  左上，右下

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i,pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text) - 1
                #scale height or width 缩放
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                bndbox.append(cur_pt)

            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res += [bndbox]   # [xmin, ymin, xmax, ymax, label_ind]

        return res   # [[xmin, ymin, xmax, ymax, label_ind], ... ]

class VOCDetection(data.Dataset):
    """
    input is image, target is annotation
    让对象实现迭代功能，返回单张图像以及其标签

    Arguments:
        root(string): filepath to VOC folder.
        image_set(string): imageset to use(eg: "train", "val", "test")
        transform(callable, optional): transformation to perform on the input_image
        target_transform (callable, optional): transformation to perform on the target `annotation`
                    (eg: take in caption string, return tensor of word indices)

        dataset_name(string, optional): which dataset to load
    """

    def __init__(self, root, image_sets=[("WasteDataset", "trainval")],transform=None, target_transform=VOCAnnotationTransform(),
                 dataset_name="WasteDataset"):

        self.root = root      #数据集的根目录
        self.image_sets = image_sets  #设置要选用的数据集，我们这里只有一个
        self.transform = transform    #定义图像转换方法
        self.target_transform = target_transform   #定义标签转换方法
        self.name = dataset_name    #定义数据集名称
        self._annopath = os.path.join("%s", 'Annotations', "%s.xml")  #匹配annotation中xml文件
        self._imgpath = os.path.join("%s", "JPEGImages", "%s.jpg")  #匹配img文件
        self.ids = list()         #记录数据集中所有图像的名称
        #推断出图片名称以及存储位置
        for (name, f) in image_sets:
            rootpath = os.path.join(self.root, name)
            for line in open(os.open.join(rootpath, "ImageSets", "Main",f + ".txt")):
                self.ids.append((rootpath, line.strip()))


    def __getitem__(self, index):
        """
        实现迭代功能，切片操作
        """
        im, gt, h, w = self.pull_item(index)

    def __len__(self):
        return len(self.ids)

    def pull_item(self, index):
        """
        对图片结合xml进行转换后返回图片及对应数据
        :param index: 图片索引
        :return: img, target, h, w
        """
        img_id = self.ids[index]

        target = ET.parse(self._annopath % img_id).getroot()  #找到对应的xml文件
        img = cv2.imread(self._imgpath % img_id) #读取对应图像
        height, width, channels = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target, width, height)  #获取objects

        if self.transform is not None:
            target = np.array(target)
            #对图像进行转换，二参为default boxes相对坐标，三参为类别
            img, boxes, labels = self.transform(img, target[:,:4], target[:,4])
            # to rgb
            img = img[:, :, (2, 1, 0)] #opencv读取图像的顺序是bgr, 转为rgb
            # img = img.transpose(2, 0, 1)
            target = np.hstack((boxes, np.expand_dims(labels, axis=1))) #将参数元组的元素数组按水平方向进行叠加

        # 返回image、label、宽高.
        # 这里的permute(2,0,1)是将原有的三维（28，28，3）变为（3，28，28），将通道数提前，为了统一torch的后续训练操作。
        return torch.from_numpy(img).permute(2, 0, 1), target, height, width

    def pull_image(self, index):
        """
        以PIL格式返回索引处的原始图像对象

        注意：不使用self .__ getitem __（），因为传入的任何转换可能会弄乱此功能。

        :argument index(int): index of img to show
        :return PIL img
        """
        img_id = self.ids[index]
        return cv2.imread(self._imgpath % img_id, cv2.IMREAD_COLOR) #全彩，忽略透明度


    def pull_anno(self, index):
        """
        Returns the original annotation of image at index

        :param index: index of img to get annotation of
        :return: list [img_id, [label, bbox coords],....]
                eg: ("001718", [('Pot', (96, 13, 438, 332))])
        """
        img_id = self.ids[index]
        anno = ET.parse(self._annopath % img_id).getroot() #Element
        gt = self.target_transform(anno, 1, 1)
        return img_id[1], gt

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

if __name__ == "__main__":
    pass

