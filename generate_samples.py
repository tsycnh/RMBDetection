from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
from rmbgenerator import RMBGenerator
import numpy as np
import cv2,os,shutil
from step7_predict import decode_predict,decode_gt
import random
data_dir = "C:\\All\\Data\\RMB\\Recognition"
train_num = 1000
val_num = 1000
test_num = 1000

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)
new_train_dir = os.path.join(data_dir,"Train")
new_val_dir = os.path.join(data_dir,"Val")
new_test_dir = os.path.join(data_dir,"Test")
os.mkdir(new_train_dir)
os.mkdir(new_val_dir)
os.mkdir(new_test_dir)

origin_data_path = "C:\\All\\Data\\RMB\\train_data"
imgs_list = os.listdir(origin_data_path)
random.shuffle(imgs_list)
model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_5-15-6.h5")
index=0
for i,name in enumerate(imgs_list):
    img_origin = load_img(os.path.join(origin_data_path,name))
    img = load_img(os.path.join(origin_data_path,name),target_size=(400,400))
    img_origin2 = img_to_array(img_origin,dtype='uint8')
    img2 = img_to_array(img,dtype='float32')
    img3 = np.expand_dims(img2/255,0)
    predict = model.predict(img3)
    x,y,w,h,c = decode_predict(predict[0])
    if c >=0.99:
        h_origin,w_origin,_ = img_origin2.shape

        w_real = w_origin*w/400
        h_real = h_origin*h/400
        x_real = w_origin*x/400
        y_real = h_origin*y/400

        crop_origin_img = img_origin2[int(y_real-h_real/2):int(y_real+h_real/2),int(x_real-w_real/2):int(x_real+w_real/2)]
        if index <train_num:
            dst_path = os.path.join(new_train_dir,name)
        elif train_num<=index<train_num+val_num:
            dst_path = os.path.join(new_val_dir,name)
        elif train_num+val_num<=index<train_num+val_num+test_num:
            dst_path = os.path.join(new_test_dir,name)
        else:
            print("Finished.")
            exit()
        cv2.imwrite(dst_path,cv2.cvtColor(crop_origin_img,cv2.COLOR_BGR2RGB))
        print("%d/%d %s"%(i+1,len(imgs_list),dst_path))
        index +=1
    else:
        print("conf too low %.2f"%c)
