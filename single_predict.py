from keras.models import load_model
from rmbgenerator import RMBGenerator
import numpy as np
import cv2
from keras.preprocessing.image import load_img, img_to_array
from keras.optimizers import rmsprop, adam, sgd
from stloss import stloss, stModelCheckpoint


# 预测单张图像
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
    return x_real, y_real, w_real, h_real, best_conf


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


def visualization(img, pd, gt, name=""):
    x_real, y_real, w_real, h_real = gt
    x_pred, y_pred, w_pred, h_pred = pd
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


def analyse_data(x_batch, y_batch, predicts):
    for i in range(y_batch.shape[0]):
        img = x_batch[i] * 255
        img = img.astype(np.uint8)
        x_real, y_real, w_real, h_real = decode_gt(y_batch[i])
        x_pred, y_pred, w_pred, h_pred, _ = decode_predict(predicts[i])

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        visualization(img, (x_pred, y_pred, w_pred, h_pred), (x_real, y_real, w_real, h_real))


if __name__ == "__main__":
    img_path = "./resource/fiveyuan.jpg"

    model = load_model("./weights/tinymind_RMBDetection.h5")
    model.compile(optimizer=rmsprop(1e-2), loss=stloss)

    img = load_img(img_path, target_size=(400, 400), interpolation="bilinear")
    img2 = img_to_array(img) / 255
    img2 = np.expand_dims(img2, 0)
    result = model.predict_on_batch(img2)
    r = decode_predict(result[0])
    print(r)
    x_pred, y_pred, w_pred, h_pred, _ = r
    grid_size = 400 / 6
    img = img_to_array(img,dtype=np.uint8)
    cv2.cvtColor(img,cv2.COLOR_RGB2BGR,img)
    for i in range(6):
        cv2.line(img, (int(grid_size) * i, 0), (int(grid_size) * i, 400 - 1), (255, 0, 0))
        cv2.line(img, (0, int(grid_size) * i), (400 - 1, int(grid_size) * i), (255, 0, 0))

    cv2.rectangle(img,(int(x_pred-w_pred/2),int(y_pred-h_pred/2)),(int(x_pred+w_pred/2),int(y_pred+h_pred/2)),color=(0,0,255))
    cv2.circle(img, (x_pred, y_pred), 1, (0, 0, 255))

    cv2.imshow("a", img)
    cv2.waitKey(0)
    pass
    # print("训练集效果")
    # analyse_data(x_batch,y_batch,predicts)
    # print("验证集效果")
    # analyse_data(x_batch_v,y_batch_v,predicts_v)
