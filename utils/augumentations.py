import torch
from torchvision import transforms
import cv2
import numpy as np
import types
from numpy import random


def intersect(box_a, box_b):
    #两个框取交集的面积,area(a ∩ b)
    #(xmin, ymin, xmax, ymax)
    max_xy = np.minimum(box_a[:,2:], box_b[2:])   #取对应位置的小值
    min_xy = np.maximum(box_a[:,:2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """
    计算两个boxes之间的jaccard overlaps(交并比，根据阈值提取匹配的boxes)
    Eg:
         A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    :param box_a: Multiple booudning boxes , Shape: [num_boxes, 4]
    :param box_b: Single bounding box, Shape: [4]
    :return:  jaccard overlap: Shape [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1]))  #[A, B]
    area_b = ((box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union

class Compose(object):
    """
    Compose several augmentations together.
    Args:
        transforms (List[Transforms]): list of transform to compose.

    Examples:
        augmentations.Compose([
             transforms.CenterCrop(10),
             transforms.ToTensor(),
        ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class Lambda(object):
    """
    Applies a lambda as a transform
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)  #判断是否为Lambda形式
        self.lambd = lambd

    def __call__(self, img, boxes=None, labels=None):
        return self.lambd(img, boxes, labels)

class ConvertFromInts(object):
    """
    数据类型转换
    """
    def __call__(self, img, boxes=None, labels=None):
        return img.astype(np.float32). boxes, labels

class SubtractMeans(object):
    """
    减去均值
    """
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, img, boxes=None, labels=None):
        img = img.astype(np.float32)
        img -= self.mean
        return img.astype(np.float32), boxes , labels

class SwapChannels(object):
    """
    Tranforms a tensroized image by swapping the channels
    Args:
        swaps(int triple): final order of channels eg:(2, 1, 0)
    """
    def __init__(self, swaps):
        self.swaps = swaps

    def __call__(self, img):
        """
        :param img: image(tensor): image tensor to be transformed
        :return: a tensor with channels swapped according to swap
        """
        img = img[:, :, self.swaps]
        return img

class ToAbsoluteCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        #还原为绝对像素坐标
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return img, boxes, labels

class ToPercentCoords(object):
    def __call__(self, img, boxes=None, labels=None):
        height, width, channels = img.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return img, boxes, labels

class Resize(object):
    def __init__(self, size=300):
        self.size = size

    def __call__(self, img, boxes=None, labels=None):
        img = cv2.resize(img, (self.size, self.size))

        return img, boxes, labels

class RandomSaturation(object):
    """
    像素内容变化，随机改变图片的饱和度
    在HSV空键的S维度上乘以一个系数（lower, upper）
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower"
        assert self.lower >=0, "contrast lower must be non-negative"

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):  # 50%几率
            img[:,:,1] *= random.uniform(self.lower, self.upper)

        return img, boxes, labels

class RandomHue(object):
    """
    随机改变色调，在HSV空间的H维度随机加一个实数
    """
    def __init__(self, delta=18.0):
        assert delta >= 0.0 and delta <= 360.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):  #50%  0 and 1
            img[:, :, 0] += random.uniform(-self.delta, self.delta)
            img[:, :, 0] [img[:, :, 0] > 360.0] -= 360.0
            img[:, :, 0][img[:, :, 0] < 0.0] += 360.0

        return img, boxes, labels

class RandomLightingNoise(object):
    """
    通过随机指定channel维度的顺序来改变图像的BGR颜色通道
    六种变换方式
    """
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            swap = self.perms[random.randint(len(self.perms))]  #随机抽取一个
            shuffle = SwapChannels(swap)
            img = shuffle(img)

        return img, boxes, labels

class ConvertColor(object):
    """
    变换颜色空间，若当前为bgr则变换到hsv,若当前为Hsv变换为bgr
    """
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform # 要变换到HSV
        self.current = current    # 当前默认为BGR

    def __call__(self, img, boxes=None, labels=None):
        if self.current == "BGR" and self.transform == "HSV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError
        return img, boxes, labels

class RandomContrast(object):
    """
    随机改变对比度，在原图像素上乘以一个系数
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
        assert self.upper >= self.lower, "contrast upper must be >= lower."
        assert self.lower >= 0, "contrast lower must be non-negative."

    # expects float image
    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            alpha = random.uniform(self.lower, self.upper)
            img *= alpha
        return img, boxes, labels

class RandomBrightness(object):
    """
    随机改变亮度，在原有图像像素上加上一个实数（实数的范围在[-32, 32]）
    """
    def __init__(self, delta=32):
        assert delta >= 0.0
        assert delta <= 255.0
        self.delta = delta

    def __call__(self, img, boxes=None, labels=None):
        if random.randint(2):
            delta = random.uniform(-self.delta, self.delta)
            img += delta
        return img, boxes, labels


class ToCV2Image(object):
    def __call__(self, tensor, boxes=None, labels=None):
        # array  bgr
        return tensor.cpu().numpy().astype(np.float32).transpose((1, 2, 0)), boxes, labels

class ToTensor(object):
    def __call__(self, cvimg, boxes=None, labels=None):
        # rgb
        return torch.from_numpy(cvimg.astype(np.float32)).permute(2, 0, 1), boxes, labels

class RandomSampleCrop(object):
    """
    随机裁剪，在图像上随机剪裁矩形区域，裁剪区域一定要包含bbox的中心点，将原始图bbox转换到剪裁区域的bbox

    Arguments:
        img (Image): the image being input during training
        boxes (Tensor): the original bounding boxes in pt form
        labels (Tensor): the class labels for each bbox
        mode (float tuple): the min and max jaccard overlaps
    Return:
        (img, boxes, classes)
            img (Image): the cropped image
            boxes (Tensor): the adjusted bounding boxes in pt form
            labels (Tensor): the class labels for each bbox
    """
    def __init__(self):
        self.sample_options = (
            # using entire original input image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )

    def __call__(self, img, boxes=None, labels=None):
        height, width, _ = img.shape
        while True:
            # randomly choose a mode
            mode = random.choice(self.sample_options)
            if mode is None:  #使用原始图
                return img, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            # max trails (50)
            for _ in range(50):
                current_img = img

                w = random.uniform(0.3 * width, width)    #裁剪的w范围
                h = random.uniform(0.3 * height, height)  #裁剪的h范围

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(width - w)
                top = random.uniform(height - h)

                # 得到裁剪图像的xmin, ymin, xmax, ymax
                rect = np.array([int(left), int(top), int(left + w), int(top + h)])

                #计算IOU,判断裁剪图像与框的
                overlap = jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # 裁剪图像
                current_img = current_img[rect[1]:rect[3], rect[0]:rect[2],:]

                # keep overlap with gt box IF center in sampled patch
                #计算中心点
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                # 检查剪裁图像的min_x, min_y要分别小于bbox的中心x, y
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                # 检查剪裁图像的max_x, max_y要分别大于bbox的中心x, y
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                # 上述两条要求都要为True
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                return current_img, current_boxes, current_labels

class Expand(object):
    def __init__(self, mean):
        self.mean = mean

    def __call__(self, img, boxes, labels):
        if random.randint(2):
            return img, boxes, labels

        height, width, depth = img.shape
        ratio = random.uniform(1, 4)
        left = random.uniform(0, width*ratio - width)
        top = random.uniform(0, height*ratio - height)

        expand_img = np.zeros(
            (int(height*ratio), int(width*ratio), depth), dtype=img.dtype)
        expand_img[:, :, :] = self.mean

        expand_img[int(top):int(top + height),
                     int(left):int(left + width)] = img
        img = expand_img

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return img, boxes, labels

class RandomMirror(object):
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

class PhotometricDistort(object):
    def __init__(self):
        self.pd = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_brightness = RandomBrightness()
        self.rand_light_noise = RandomLightingNoise()

    def __call__(self, image, boxes, labels):
        im = image.copy()
        im, boxes, labels = self.rand_brightness(im, boxes, labels)
        if random.randint(2):
            distort = Compose(self.pd[:-1])
        else:
            distort = Compose(self.pd[1:])
        im, boxes, labels = distort(im, boxes, labels)
        return self.rand_light_noise(im, boxes, labels)

class SSDAugmentation(object):
    def __init__(self, size=300, mean=(104, 117, 123)):
        self.mean = mean
        self.size = size
        self.augment = Compose([
            ConvertFromInts(),  # 数据类型转换
            ToAbsoluteCoords(),  # 位置信息转换
            PhotometricDistort(),  # 镜像翻转
            Expand(self.mean),  # 扩展图像
            RandomSampleCrop(),  # 随机裁剪
            RandomMirror(),  # 随机镜像翻转
            ToPercentCoords(),  # 位置归一化
            Resize(self.size),  # 图像尺寸缩放
            SubtractMeans(self.mean)  # 图像去均值
        ])

    def __call__(self, img, boxes, labels):
        return self.augment(img, boxes, labels)

if __name__ == "__main__":
    pass