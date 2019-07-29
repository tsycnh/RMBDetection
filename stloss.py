import numpy as np
import tensorflow as tf
import warnings
from keras.backend import eval
from keras.callbacks import Callback
class stModelCheckpoint(Callback):
    """
    修改原始检查点保存函数，不保存optimizer信息
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(stModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True,include_optimizer=False)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True,include_optimizer=False)

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
    loss = 10*loss_obj + 5 * loss_noobj + 5 * loss_coord
    return loss

if __name__ == "__main__":
    a = np.load("y_batch.npy")
    b = np.random.rand(4, 6, 6, 5).astype('float32')
    ta = tf.constant(a)
    tb = tf.constant(b)
    loss =stloss(ta, tb)
    print(eval(loss))