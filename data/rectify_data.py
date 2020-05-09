import os
import re

if __name__  == "__main__":
    f = open(r"E:\WasteDataset\name.txt")

    new_pattern = r"\d{6}\.xml"
    origin_pattern = r"\d{6}\.jpg"

    transfer = {}

    for line in f.readlines():
        print(line)
        if line.strip() == "":
            break
        new_name = re.search(new_pattern, line).group()
        old_name = re.search(origin_pattern, line).group()

        transfer[old_name] = new_name
        print(old_name+" : "+ new_name)

    print(len(transfer))


    filepath = "E:\\WasteDataset\\Annotations"
    filelist = os.listdir(filepath)

    filelist.sort(key = lambda x : int(x[:-4]))

    for item in filelist:

        # 修改错误后缀名以及序号
        if item.endswith(".jpg") and item in transfer.keys():
            src = os.path.join(os.path.abspath(filepath), item)
            dst = os.path.join(os.path.abspath(filepath), transfer[item])
            os.rename(src, dst)
            print(src + " --> " + dst)


        # 删除标注不合格的图片
        if int(item[:-4]) in [i for i in range(4302, 7379)]:
            src = os.path.join(os.path.abspath(filepath), item)
            os.remove(src)



