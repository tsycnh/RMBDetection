from keras.models import load_model
from rmbgenerator import RMBGenerator
import numpy as np
import cv2
# model = load_model("C:\\All\\Tdevelop\\RMBDetection\\tmp\\RMBdt_weights_19_valloss_0.56.h5")
model = load_model("C:\\All\\Tdevelop\\RMBDetection\\tmp\\RMBdt_weights_17_valloss_4.17.h5")
batch_size = 8
train_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\train\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\train\\annos",
                         batch_size=batch_size,rescale=1.0/255)
train_gen.on_epoch_end()

x_batch,y_batch = train_gen.__getitem__(0)

p = model.predict(x_batch)
for i in range(batch_size):
    img = x_batch[i] * 255
    img = img.astype(np.uint8)
    def cvt_coord(raw):
        c_x, c_y, w, h = raw * 400
        return (int(c_x - w / 2), int(c_y - h / 2)), (int(c_x + w / 2), int(c_y + h / 2))
    gt = cvt_coord(y_batch[i])
    pd = cvt_coord(p[i])
    c_x, c_y, w, h = y_batch[0] * 400
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.rectangle(img, gt[0], gt[1], color=(0, 255, 0),thickness=2)
    cv2.rectangle(img, pd[0], pd[1], color=(0, 0, 255),thickness=2)
    cv2.imshow(str(i), img)
    cv2.waitKey(0)
