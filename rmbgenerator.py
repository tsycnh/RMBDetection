from keras.utils import Sequence
from keras.preprocessing.image import load_img,img_to_array
import os,shutil
import numpy as np
from pascal_voc_tools import XmlParser
import cv2
import random
import imgaug.augmenters as iaa
import imgaug.imgaug as ia
from imgaug.augmentables.kps import Keypoint, KeypointsOnImage


# 自定义一个数据生成器。继承自keras.utils.Sequence,可以和fit_generator无缝衔接
# Sequence 是keras中数据生成器的基类。必须实现__getitem__(),__len__(),建议实现on_epoch_end(),可以在这里打乱数据集
class RMBGenerator(Sequence):
    def __init__(self, images_dir, annos_dir, batch_size,rescale=1,aug=False):
        self.batch_size = batch_size
        self.images_dir = images_dir
        self.annos_dir = annos_dir
        self.rescale = rescale
        images_list = os.listdir(images_dir)
        annos_list = os.listdir(annos_dir)
        images_list.sort()
        annos_list.sort()
        self.pair = list(zip(images_list,annos_list))
        self.aug = aug
        if self.aug:
            self.seq_o = iaa.Sequential([iaa.Affine(translate_px={"x": (-30, 30), "y": (-30, 30)})])
            self.seq = self.seq_o.to_deterministic()
            # self.seq = iaa.Sequential([iaa.Affine(translate_px=16)])

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
                img = load_img(os.path.join(self.images_dir,p[0]),target_size=(400,400),interpolation="bilinear")
                anno = XmlParser().load(os.path.join(self.annos_dir,p[1]))

                coded_img = self.code_img(img)
                coded_anno = self.code_anno(anno)

                x_batch[i] = coded_img
                y_batch[i] = coded_anno
                # print(p[0],p[1])
        return x_batch*self.rescale,y_batch
    def on_epoch_end(self):
        random.shuffle(self.pair)
        pass
    def code_img(self,img):
        coded_img = img_to_array(img)
        if self.aug:
            coded_img = self.seq.augment_image(coded_img)
        return coded_img
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

        if self.aug:
            d= KeypointsOnImage([Keypoint(c_x*400, c_y*400)], (400, 400))
            fuck = self.seq.augment_keypoints([d])[0]
            c_x = fuck.keypoints[0].x/400
            c_y = fuck.keypoints[0].y/400

        grid_x = int(np.floor(c_x*6))
        grid_y = int(np.floor(c_y*6))
        relative_x = c_x*6 - grid_x
        relative_y = c_y*6 - grid_y

        coded_anno[grid_y,grid_x,:] = np.array([1,relative_x,relative_y,w,h])
        return coded_anno

if __name__ == "__main__":
    # 检测输出
    batch_size = 32
    rmbd = RMBGenerator("C:\\All\\Data\\RMB\\Detection\\train\\images",
                        "C:\\All\\Data\\RMB\\Detection\\train\\annos",
                        batch_size,rescale=1./255,aug=True)
    # rmbd.on_epoch_end()
    x_batch,y_batch = rmbd.__getitem__(0)
    for i in range(batch_size):
        img = x_batch[i]*255
        img = img.astype(np.uint8)
        np.save("y_batch.np",y_batch)
        # f =np.argmax(y_batch[0])
        mat = y_batch[i]
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
        cv2.imshow("demo",img)
        cv2.waitKey(0)
