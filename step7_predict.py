from keras.models import load_model
from rmbgenerator import RMBGenerator
import numpy as np
import cv2

def decode_predict(predict):
    pd = predict.copy()
    conf = pd[..., 0]
    pos = pd[..., 1:]
    conf[conf < 0.5] = 0
    y_grid_pd, x_grid_pd = np.unravel_index(np.argmax(conf), conf.shape)
    best_conf = conf[y_grid_pd, x_grid_pd]
    print("pd:y_grid:%d x_grid:%d conf:%.2f" % (y_grid_pd, x_grid_pd, best_conf))

    xywh = pos[y_grid_pd, x_grid_pd, :]
    x_real = int((400 / 6) * (x_grid_pd + xywh[0]))
    y_real = int((400 / 6) * (y_grid_pd + xywh[1]))
    w_real = int(400 * xywh[2])
    h_real = int(400 * xywh[3])
    return x_real, y_real, w_real, h_real,best_conf

def decode_gt(gt):
    conf = gt[..., 0]
    y_grid, x_grid = np.unravel_index(np.argmax(conf), conf.shape)
    print("gt:y_grid:%d x_grid:%d" % (y_grid, x_grid))
    _, x_rela, y_rela, w, h = gt[y_grid][x_grid][:].tolist()
    x_real = int((400 / 6) * (x_grid + x_rela))
    y_real = int((400 / 6) * (y_grid + y_rela))
    w_real = int(400 * w)
    h_real = int(400 * h)
    return x_real, y_real, w_real, h_real

def visualization(img,pd,gt,name=""):
    x_real,y_real,w_real,h_real = gt
    x_pred,y_pred,w_pred,h_pred = pd
    grid_size = 400 / 6
    for i in range(6):
        cv2.line(img, (int(grid_size) * i, 0), (int(grid_size) * i, 400 - 1), (255, 0, 0))
        cv2.line(img, (0, int(grid_size) * i), (400 - 1, int(grid_size) * i), (255, 0, 0))

    cv2.rectangle(img, (int(x_real - w_real / 2), int(y_real - h_real / 2)),
                  (int(x_real + w_real / 2), int(y_real + h_real / 2)), color=(0, 255, 0), thickness=1)
    cv2.circle(img, (x_real, y_real), 1, (0, 255, 0))
    cv2.rectangle(img, (int(x_pred - w_pred / 2), int(y_pred - h_pred / 2)),
                  (int(x_pred + w_pred / 2), int(y_pred + h_pred / 2)), color=(0, 0, 255), thickness=1)
    cv2.circle(img, (x_pred, y_pred), 1, (0, 0, 255))

    cv2.imshow(name, img)
    cv2.waitKey(0)
def analyse_data(x_batch,y_batch,predicts):
    for i in range(y_batch.shape[0]):
        img = x_batch[i] * 255
        img = img.astype(np.uint8)
        x_real,y_real,w_real,h_real = decode_gt(y_batch[i])
        x_pred,y_pred,w_pred,h_pred,_ = decode_predict(predicts[i])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualization(img,(x_pred,y_pred,w_pred,h_pred),(x_real,y_real,w_real,h_real))


if __name__ == "__main__":
    batch_size = 5
    train_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\train\\images",
                             annos_dir="C:\\All\\Data\\RMB\\Detection\\train\\annos",
                             batch_size=batch_size, rescale=1.0 / 255)
    val_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\val\\images",
                           annos_dir="C:\\All\\Data\\RMB\\Detection\\val\\annos",
                           batch_size=batch_size, rescale=1.0 / 255)
    model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_5-15-6.h5")

    x_batch, y_batch = train_gen.__getitem__(0)
    x_batch_v, y_batch_v = val_gen.__getitem__(0)
    predicts = model.predict_on_batch(x_batch)
    predicts_v = model.predict_on_batch(x_batch_v)
    # np.save('tmp/predicts',predicts)
    # np.save('tmp/x_batch',x_batch)
    # np.save('tmp/y_batch',y_batch)
    # # exit()
    # import numpy as np
    # import cv2
    # x_batch = np.load('./tmp/x_batch.npy')
    # y_batch = np.load('./tmp/y_batch.npy')
    # predicts= np.load('./tmp/predicts.npy')
    # predicts:4x6x6x5
    print("训练集效果")
    analyse_data(x_batch,y_batch,predicts)
    print("验证集效果")
    analyse_data(x_batch_v,y_batch_v,predicts_v)
