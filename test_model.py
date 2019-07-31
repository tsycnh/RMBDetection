from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from rmbgenerator import RMBGenerator
import numpy as np
import cv2,os,shutil
from step7_predict import decode_predict,decode_gt
import random

data_dir = "C:\\All\\Tdevelop\\RMBDetection\\resource\\testdata"
imgs_list = os.listdir(data_dir)
model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_5-11.h5")

for i,name in enumerate(imgs_list):
    img_path = os.path.join(data_dir,name)
    img_origin = load_img(img_path)
    img = load_img(img_path,target_size=(400,400))
    img_origin2 = img_to_array(img_origin,dtype='uint8')
    img2 = img_to_array(img,dtype='float32')
    img3 = np.expand_dims(img2/255,0)
    predict = model.predict(img3)
    x,y,w,h = decode_predict(predict[0])
    h_origin,w_origin,_ = img_origin2.shape

    w_real = w_origin*w/400
    h_real = h_origin*h/400
    x_real = w_origin*x/400
    y_real = h_origin*y/400

    crop_origin_img = img_origin2[int(y_real-h_real/2):int(y_real+h_real/2),
                                  int(x_real-w_real/2):int(x_real+1.2*w_real/2)]

    img = img2.astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow('small_clean',img)
    grid_size = 400 / 6
    for i in range(6):
        cv2.line(img, (int(grid_size) * i, 0), (int(grid_size) * i, 400 - 1), (255, 0, 0))
        cv2.line(img, (0, int(grid_size) * i), (400 - 1, int(grid_size) * i), (255, 0, 0))

    cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)),
                  (int(x + w / 2), int(y + h / 2)), color=(0, 255, 0), thickness=1)
    cv2.circle(img, (x, y), 1, (0, 255, 0))
    cv2.imshow('small',img)
    cv2.waitKey(0)
    # 绘制原始图像大小
    img_origin3 = cv2.cvtColor(img_origin2, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img_origin3, (int(x_real - w_real / 2), int(y_real - h_real / 2)),
                  (int(x_real + w_real / 2), int(y_real + h_real / 2)), color=(0, 0, 255), thickness=2)

    cv2.imshow("origin",img_origin3)
    cv2.waitKey(0)

#
# x_batch, y_batch = train_gen.__getitem__(0)
# x_batch_v, y_batch_v = val_gen.__getitem__(0)
# predicts = model.predict_on_batch(x_batch)
# predicts_v = model.predict_on_batch(x_batch_v)
#
#
def analyse_data(x_batch,y_batch,predicts):
    for i in range(y_batch.shape[0]):
        img = x_batch[i] * 255
        img = img.astype(np.uint8)
        x_real,y_real,w_real,h_real = decode_gt(y_batch[i])
        x_pred,y_pred,w_pred,h_pred = decode_predict(predicts[i])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        cv2.rectangle(img, (int(x_real - w_real / 2), int(y_real - h_real / 2)),
                      (int(x_real + w_real / 2), int(y_real + h_real / 2)), color=(0, 255, 0), thickness=2)
        cv2.rectangle(img, (int(x_pred - w_pred / 2), int(y_pred - h_pred / 2)),
                      (int(x_pred + w_pred / 2), int(y_pred + h_pred / 2)), color=(255, 0, 0), thickness=2)
        cv2.imshow(str(i), img)
        cv2.waitKey(0)
#
# analyse_data(x_batch,y_batch,predicts)
# analyse_data(x_batch_v,y_batch_v,predicts_v)
