from keras.utils import Sequence
from keras.preprocessing.image import load_img,img_to_array
import os,shutil
import numpy as np
from pascal_voc_tools import XmlParser
import cv2
import random
'''
生成器设计：
输入一个batch的图像，如:32x400x400x3
输出：32x4
其中4代表着图像中唯一的目标框。分别为center_x,center_y,w,h。
前两者为将图像长宽归一化后目标框中心点相对于左上角的坐标
后两者为目标框相对于归一化目标框的大小。
4个值取值范围都是0~1          
'''
# 自定义一个数据生成器。继承自keras.utils.Sequence,可以和fit_generator无缝衔接
# Sequence 是keras中数据生成器的基类。必须实现__getitem__(),__len__(),建议实现on_epoch_end(),可以在这里打乱数据集
class RMBGenerator(Sequence):
    def __init__(self, images_dir, annos_dir, batch_size,rescale=1):
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.annos_dir = annos_dir
        self.rescale = rescale
        images_list = os.listdir(images_dir)
        annos_list = os.listdir(annos_dir)
        images_list.sort()
        annos_list.sort()
        self.pair = list(zip(images_list,annos_list))

    def __len__(self):
        return int(np.ceil(len(self.pair) / float(self.batch_size)))

    def __getitem__(self, idx):
        start_index = idx*self.batch_size
        stop_index = (idx+1)*self.batch_size
        x_batch = np.zeros((self.batch_size,400,400,3),dtype='float32')
        y_batch = np.zeros((self.batch_size,4),dtype='float32')
        for i,p in enumerate(self.pair[start_index:stop_index]):
            if p[0].split(".")[0] != p[1].split(".")[0]:
                print("图像名与标记名不符！%s %s"%(p[0],p[1]))
                exit()
            else:
                img = load_img(os.path.join(self.images_dir,p[0]),target_size=(400,400))
                anno = XmlParser().load(os.path.join(self.annos_dir,p[1]))

                coded_img = img_to_array(img)
                coded_anno = self.code_anno(anno)
                x_batch[i] = coded_img
                y_batch[i] = coded_anno
                # print(p[0],p[1])
        return x_batch*self.rescale,y_batch
    def on_epoch_end(self):
        random.shuffle(self.pair)
        pass
    def code_anno(self,anno):
        origin_w = int(anno['size']['width'])
        origin_h = int(anno['size']['height'])

        xmin = int(anno['object'][0]['bndbox']['xmin']) -1
        ymin = int(anno['object'][0]['bndbox']['ymin']) -1
        xmax = int(anno['object'][0]['bndbox']['xmax']) -1
        ymax = int(anno['object'][0]['bndbox']['ymax']) -1

        return np.array([
            ((xmax+xmin)/2)/origin_w,
            ((ymax+ymin)/2)/origin_h,
            (xmax-xmin)/origin_w,
            (ymax-ymin)/origin_h
        ])

if __name__ == "__main__":
    # 检测输出
    rmbd = RMBGenerator("C:\\All\\Data\\RMB\\Detection\\train\\images",
                        "C:\\All\\Data\\RMB\\Detection\\train\\annos",
                        4,rescale=1./255)
    rmbd.on_epoch_end()
    x_batch,y_batch = rmbd.__getitem__(0)

    img = x_batch[0]*255
    img = img.astype(np.uint8)
    c_x,c_y,w,h = y_batch[0]*400
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.rectangle(img,(int(c_x-w/2),int(c_y-h/2)),(int(c_x+w/2),int(c_y+h/2)),color=(0,255,0),thickness=2)
    cv2.imshow("f",img)
    cv2.waitKey(0)
