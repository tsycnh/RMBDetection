import numpy as np
import numpy.ma as ma
import tensorflow as tf

def stloss(y_true,y_pred):

    # 真值
    conf = y_true[...,0]
    conf = np.expand_dims(conf,-1)
    pos  = y_true[...,1:]
    # 预测值
    conf_pred = y_pred[...,0]
    conf_pred = np.expand_dims(conf_pred,-1)
    pos_pred  = y_pred[...,1:]


    noobj_conf = ma.masked_equal(conf,1)   #掩盖存在物体的grid
    obj_conf = ma.masked_equal(conf,0)     #掩盖不存在物体的grid
    obj_pos = ma.masked_equal(pos,0)

    loss_obj = np.sum(np.square(obj_conf - conf_pred))
    loss_noobj = np.sum(np.square(noobj_conf - conf_pred))
    loss_coord = np.sum(np.square(obj_pos - pos_pred))

    loss = (loss_obj+0.5*loss_noobj+5*loss_coord)/tf.cast(tf.shape(y_true)[0],tf.float32)
    return loss

if __name__ == "__main__":
     a = np.load("y_batch.npy")
     b = np.random.rand(4,6,6,5).astype('float32')
     ta = tf.constant(a)
     tb = tf.constant(b)
     stloss(ta,tb)
