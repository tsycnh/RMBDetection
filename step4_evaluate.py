# 上测试集

from keras.models import load_model
from keras.preprocessing.image import load_img,img_to_array
import time,os
import numpy as np
f = open("./test_result.csv",mode='w')
f.write("name,label\n")
model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_3-1.h5")
class_indices = {'0.1': 0, '0.2': 1, '0.5': 2, '1': 3, '10': 4, '100': 5, '2': 6, '5': 7, '50': 8}
index = {v:k for k,v in class_indices.items()}
print(index)
# 2. prepare data
public_test_dir = "C:\\All\\Data\\RMB\\public_test_data"
imgs = os.listdir(public_test_dir)

for i,img in enumerate(imgs):
    img_path = os.path.join(public_test_dir,img)
    image = load_img(img_path,target_size=(400,400))
    image2 = img_to_array(image)
    image3 = image2/255
    image4 = np.expand_dims(image3,axis=0)
    result = model.predict(image4)
    m = np.argmax(result)
    # print(index[m])
    # image.show()
    line = img+", "+index[m]+'\n'
    f.write(line)
    print("%d/%d %s"%(i+1,len(imgs),line))

f.close()