import numpy as np
import numpy.ma as ma
import tensorflow as tf
from keras.backend import eval


def stloss(y_true, y_pred):
    # 真值
    conf_true = y_true[..., 0]
    pos_true = y_true[..., 1:]
    # 预测值
    conf_pred = y_pred[..., 0]
    pos_pred = y_pred[..., 1:]
    # indices0,1: 置信度为0,1的值的坐标  indices1_pos: 不为空的bbox的xywh的坐标
    indices0 = tf.where(tf.equal(conf_true, tf.zeros([1.])))
    indices1 = tf.where(tf.equal(conf_true, tf.ones([1.])))
    indices1_pos = tf.where(tf.greater(pos_true, tf.zeros([1.])))
    # 依据indices提取相应的置信度值并组成新的向量
    conf_true_0 = tf.gather_nd(conf_true, indices0)
    conf_true_1 = tf.gather_nd(conf_true, indices1)
    conf_pred_0 = tf.gather_nd(conf_pred, indices0)
    conf_pred_1 = tf.gather_nd(conf_pred, indices1)
    # 依据indices提取相应xywh坐标并组成新的向量
    pos_true1 = tf.gather_nd(pos_true, indices1_pos)
    pos_pred1 = tf.gather_nd(pos_pred, indices1_pos)
    # 分配计算loss
    loss_noobj = tf.losses.mean_squared_error(conf_true_0, conf_pred_0)
    loss_obj = tf.losses.mean_squared_error(conf_true_1, conf_pred_1)
    loss_coord = tf.losses.mean_squared_error(pos_true1, pos_pred1)
    # 加权
    loss = loss_obj + 0.5 * loss_noobj + 5 * loss_coord
    return loss


if __name__ == "__main__":
    a = np.load("y_batch.npy")
    b = np.random.rand(4, 6, 6, 5).astype('float32')
    ta = tf.constant(a)
    tb = tf.constant(b)
    loss =stloss(ta, tb)
    print(eval(loss))
