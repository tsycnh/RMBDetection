import os,shutil
origin_train_dir = "C:\\All\\Data\\RMB\\train_data"
anno_path = "C:\\All\\Tdevelop\\RMBDetection\\resource\\VOC2007"
new_dir = "C:\\All\\Data\\RMB\\Detection"
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
os.mkdir(new_dir)

paths =[]
for a in ["train","val"]:
    for b in ['images','annos']:
        paths.append(os.path.join(new_dir,a,b))
        os.makedirs(os.path.join(new_dir,a,b))

f_train = open(os.path.join(anno_path,"ImageSets","Main","train.txt"))
f_train_list = f_train.read().splitlines()
f_train.close()
f_val = open(os.path.join(anno_path,"ImageSets","Main","val.txt"))
f_val_list = f_val.read().splitlines()
f_val.close()
for i,phase in enumerate([f_train_list,f_val_list]):
    for name in phase:
        origin_image_path = os.path.join(origin_train_dir,name+".jpg")
        origin_anno_path = os.path.join(anno_path,"Annotations",name+".xml")
        dst_image_path = os.path.join(paths[0+2*i],name+".jpg")
        dst_anno_path = os.path.join(paths[1+2*i],name+".xml")
        shutil.copy(origin_image_path,dst_image_path)
        shutil.copy(origin_anno_path,dst_anno_path)
        print(dst_image_path)










