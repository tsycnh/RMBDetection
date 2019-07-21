'''
把训练集和验证集每类各拿出来100张组合成小规模数据集
'''
import os,shutil

dir = "C:\\All\\Data\\RMB\\NEW"
train_dir = os.path.join(dir,'Train')
val_dir = os.path.join(dir,'Val')
new_dir = "C:\\All\\Data\\RMB\\NEW_MINI"
new_train_dir = os.path.join(new_dir,"Train")
new_val_dir = os.path.join(new_dir,"Val")

if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.mkdir(new_dir)
for new_dir in [new_train_dir,new_val_dir]:
    os.mkdir(new_dir)
    os.mkdir(os.path.join(new_dir,"0.1"))
    os.mkdir(os.path.join(new_dir,"0.2"))
    os.mkdir(os.path.join(new_dir,"0.5"))
    os.mkdir(os.path.join(new_dir,"1"))
    os.mkdir(os.path.join(new_dir,"2"))
    os.mkdir(os.path.join(new_dir,"5"))
    os.mkdir(os.path.join(new_dir,"10"))
    os.mkdir(os.path.join(new_dir,"50"))
    os.mkdir(os.path.join(new_dir,"100"))


def copy_imgs(dir,new_dir):
    dir_list = os.listdir(train_dir)
    for sub_dir in dir_list:
        full_sub_dir = os.path.join(dir,sub_dir)
        img_list = os.listdir(full_sub_dir)
        for i,img_name in enumerate(img_list):
            img_full_path = os.path.join(full_sub_dir,img_name)
            dst_path = os.path.join(new_dir,sub_dir,img_name)
            shutil.copy(img_full_path,dst_path)
            print(dst_path)
            if i == 99:
                break
copy_imgs(train_dir,new_train_dir)
copy_imgs(val_dir,new_val_dir)