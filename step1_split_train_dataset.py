import os,shutil
import pandas as pd
data_dir = "C:\\All\\Data\\RMB"
train_dir = os.path.join(data_dir,"train_data")
face_label_fname = os.path.join(data_dir,"train_face_value_label.csv")
train_count = 20000

# if os.path.exists(os.path.join(data_dir,"Train")):
#     shutil.rmtree(os.path.join(data_dir,"Train"))
# if os.path.exists(os.path.join(data_dir, "Val")):
#     shutil.rmtree(os.path.join(data_dir,"Val"))

new_train_dir = os.path.join(data_dir,"Train")
new_val_dir = os.path.join(data_dir,"Val")
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

train_data = pd.read_csv(face_label_fname,dtype=str)
for i in range(train_data.shape[0]):
    img_name,value = train_data.loc[i].tolist()
    ori_img_path = os.path.join(train_dir,img_name)
    if i<train_count:
        dst_img_path = os.path.join(new_train_dir,value.strip(),img_name)
    else:
        dst_img_path = os.path.join(new_val_dir,value.strip(),img_name)

    shutil.copy(ori_img_path,dst_img_path)
    print("已拷贝：%d/%d %s"%(i+1,train_data.shape[0],dst_img_path))