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
        y_batch = np.zeros((self.batch_size,6,6,5),dtype='float32')
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
        coded_anno = np.zeros((6,6,5),dtype='float32')
        origin_w = int(anno['size']['width'])
        origin_h = int(anno['size']['height'])

        xmin = int(anno['object'][0]['bndbox']['xmin']) -1
        ymin = int(anno['object'][0]['bndbox']['ymin']) -1
        xmax = int(anno['object'][0]['bndbox']['xmax']) -1
        ymax = int(anno['object'][0]['bndbox']['ymax']) -1

        c_x =   ((xmax+xmin)/2)/origin_w
        c_y =   ((ymax+ymin)/2)/origin_h
        w   =   (xmax-xmin)/origin_w
        h   =   (ymax-ymin)/origin_h

        grid_x = int(np.floor(c_x*6))
        grid_y = int(np.floor(c_y*6))
        relative_x = c_x*6 - grid_x
        relative_y = c_y*6 - grid_y

        coded_anno[grid_y,grid_x,:] = np.array([1,relative_x,relative_y,w,h])
        return coded_anno

if __name__ == "__main__":
    # 检测输出
    rmbd = RMBGenerator("C:\\All\\Data\\RMB\\Detection\\train\\images",
                        "C:\\All\\Data\\RMB\\Detection\\train\\annos",
                        4,rescale=1./255)
    # rmbd.on_epoch_end()
    x_batch,y_batch = rmbd.__getitem__(0)

    img = x_batch[0]*255
    img = img.astype(np.uint8)
    np.save("y_batch.np",y_batch)
    # f =np.argmax(y_batch[0])
    mat = y_batch[0]
    a = np.argmax(mat)
    b = mat.shape
    y_grid,x_grid,_ =np.unravel_index(np.argmax(mat),mat.shape)
    _,x_rela,y_rela,w,h = mat[y_grid][x_grid][:].tolist()
    x_real =int((400/6)*(x_grid+x_rela))
    y_real =int((400/6)*(y_grid+y_rela))
    w_real = int(400*w)
    h_real = int(400*h)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    cv2.rectangle(img,(int(x_real-w_real/2),int(y_real-h_real/2)),(int(x_real+w_real/2),int(y_real+h_real/2)),color=(0,255,0),thickness=2)
    cv2.imshow("f",img)
    cv2.waitKey(0)
