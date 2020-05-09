import os
import random
import sys
if sys.version_info[0] == 2: #sys.version_info(major=3, minor=6, micro=2, releaselevel='final', serial=0)
    import xml.etree.cElementTree as ET   #解析xml文件
else:
    import xml.etree.ElementTree as ET

VOC_CLASSES = (
    'DisposableFastFoodBox', 'StainedPlastic', 'CigaretteButts', 'ToothPick', 'Dishes',
    'BambooChopsticks', 'Peel', 'Eggshell', 'PowerBank', 'Backpack', 'CosmeticBottle', 'PlasticToys',
    'PlasticTableware', 'PlasticHangers', 'OldClothers', 'Cans', 'Pillow', 'StuffedToy', 'GlassBottle',
    'LeatherShoes', 'ChoppingBoard', 'CardboardBox', 'WineBottles', 'MetalFoodCan', 'Pot', 'DryCell',
    'Ointment', 'ExpiredDrugs'
)

def generate_train_val_test_txt(path):
    xmlfilepath = os.path.join(os.path.abspath(path), "Annotations")
    txtsavepath = os.path.join(os.path.abspath(path), "ImageSets\\Main")

    ###########################################
    trainval_percent = 0.9
    train_percent = 0.8

    total_xml = os.listdir(xmlfilepath)   #得到文件夹下所有文件名称
    list = range(len(total_xml))
    tv = int(len(total_xml) * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)
    print("train and val size", tv)
    print("train size", tr)

    ###########################################
    """
    将信息写入test.txt、train.txt、val.txt、trainval.txt
    """
    ftrainval = open(os.path.join(txtsavepath, "trainval.txt"), "w")
    ftest = open(os.path.join(txtsavepath, "test.txt"), "w")
    ftrain = open(os.path.join(txtsavepath, "train.txt"), "w")
    fval = open(os.path.join(txtsavepath, "val.txt"), "w")

    for i in list:
        xml_name = total_xml[i][:-4]  #截出数值
        if i in trainval:
            ftrainval.write(xml_name + "\n")
            if i in train:
                ftrain.write(xml_name + "\n")
            else:
                fval.write(xml_name + "\n")
        else:
            ftest.write(xml_name + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

    ##########################################
    """
    写入(class_name)_test.txt....
    """
    for idx in range(len(VOC_CLASSES)):   #每个类需要单独处理
        class_name = VOC_CLASSES[idx]

        #创建txt
        class_ftrainval = open(os.path.join(txtsavepath, str(class_name)+"_trainval.txt"), "w")
        class_ftest = open(os.path.join(txtsavepath, str(class_name)+"_test.txt"), "w")
        class_ftrain = open(os.path.join(txtsavepath, str(class_name)+"_train.txt"), "w")
        class_fval = open(os.path.join(txtsavepath, str(class_name)+"_val.txt"), "w")

        positive_num = 0
        negative_num = 0

        for k in list:
            xml_name = total_xml[k][:-4]  #从xml中筛选出类别信息来分类

            xml_path = os.path.join(xmlfilepath, xml_name+".xml")

            #将获取的xml进行解析
            tree = ET.parse(xml_path)
            root = tree.getroot()

            #获取xml object的name标签
            object_name =[obj.find('name').text.lower().strip() for obj in root.iter("object")]
            # 存在object（矩形框并且class_name在object_name列表
            if len(object_name) > 0 and str(class_name).lower().strip() in object_name:
                positive_num += 1
                if k in trainval:
                    class_ftrainval.write(xml_name + " " + str(1)+ "\n")
                    if k in train:
                        class_ftrain.write(xml_name + " " + str(1) + "\n")
                    else:
                        class_fval.write(xml_name + " " + str(1) + "\n")
                else:
                    class_ftest.write(xml_name + " " + str(1) + "\n")

            else:
                negative_num += 1
                if k in trainval:
                    class_ftrainval.write(xml_name + " " + str(-1) + "\n")
                    if k in train:
                        class_ftrain.write(xml_name + " " + str(-1) + "\n")
                    else:
                        class_fval.write(xml_name + " " + str(-1) + "\n")
                else:
                    class_ftest.write(xml_name + " " + str(-1) + "\n")

        class_ftrainval.close()
        class_ftrain.close()
        class_ftest.close()
        class_fval.close()

        print( "%s have %d +1 and %d -1"%(class_name, positive_num, negative_num))

if __name__ == "__main__":
    generate_train_val_test_txt("E:\\WasteDataset")