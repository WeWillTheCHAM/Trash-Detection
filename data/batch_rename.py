import os

class BatchRename():
    '''
    批量重命名文件夹中的图片文件
    '''

    def __init__(self):
        #图片的地址
        pass

    def rename_and_clean(self, filepath, type):
        #返回文件夹中包含的文件或文件夹的名字的列表。
        filelist = os.listdir(filepath)

        #按照原有的序号排序,不符合排序方式的名称都往后放
        filelist.sort(key = lambda x : int(x[:-4]) if str(x[:-4]).isdigit() else 100000)

        total_num = len(filelist)

        i = 0
        n = 6
        for item in filelist:
            if item.endswith(type):
                n = 6 - len(str(i))  #共六位，n代表0的个数
                src = os.path.join(os.path.abspath(filepath), item)
                dst = os.path.join(os.path.abspath(filepath), str(0)*n +str(i) + '.'+type)

                try:
                    os.rename(src, dst)
                    print('coverting %s to %s...'%(src, dst))
                    i = i + 1
                except:
                    print("error")
                    continue

            else:
                #删除没用的文件
                src = os.path.join(os.path.abspath(filepath), item)
                os.remove(src)

        print("total %d to rename & converted %d %ss"% (total_num, i, type))


    def select_modest_data(self, xmlpath, datapath):
        '''
        将xml文件名统计，删除没有xml对应文件的图片
        :return:
        '''

        xmllist = os.listdir(xmlpath)
        datalist = os.listdir(datapath)

        reslist = []

        for item in xmllist:
            if item.endswith(".xml"):
                reslist.append(item[:-4])

        #按照序号排好
        reslist.sort()
        print(" total: %d and the first one is %s" % (len(reslist), reslist[0]))

        #删除不对应的图片
        for item in datalist:
            if item.endswith(".jpg") and item[:-4] in reslist:
                print(item + " reserved")

            else:
                print(item + " deleted")

                #删除没用的文件
                src = os.path.join(os.path.abspath(datapath), item)
                os.remove(src)

if __name__ == "__main__":
    datapath = "E:\\WasteDataset\\JPEGImages"
    xmlpath = "E:\\WasteDataset\\Annotations"

    demo = BatchRename()
    demo.select_modest_data(xmlpath, datapath)

    demo.rename_and_clean(xmlpath, "xml")
    demo.rename_and_clean(datapath, "jpg")


