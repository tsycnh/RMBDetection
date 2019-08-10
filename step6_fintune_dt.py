from keras.models import Sequential,load_model
from keras import losses
from keras.callbacks import TensorBoard,TerminateOnNaN,LearningRateScheduler
from keras.optimizers import rmsprop
from keras.utils import plot_model
import time,os
from rmbgenerator import RMBGenerator
from stloss import stloss,stModelCheckpoint
from utils import ring

model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\RMBdt_weights_08_loss_1.316_valloss_3.501.h5")
learning_rate = 1e-3
warmup = 3
warmup_lr = 1e-5
epochs = 30
gap_layer = "activation_46"# activation_46:解锁最后一组模块 activation_43:解锁最后两组模块
data_aug = False
#-------------------
plot_model(model.layers[0])# 绘制主干模型
# exit()
trainable = False
for layer in model.layers[0].layers:
    if trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    if layer.name == gap_layer:
        trainable = True
model.compile(optimizer=rmsprop(lr=learning_rate),loss=stloss)
model.summary()

# 2. prepare data
# 准备数据
train_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\train\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\train\\annos",
                         batch_size=32,rescale=1.0/255,aug=data_aug)
val_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\val\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\val\\annos",
                         batch_size=4,rescale=1.0/255)
format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

def lr_schedual(epoch,lr):
    if epoch < warmup:
        return warmup_lr
    else:
        return learning_rate
callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             # LearningRateScheduler(schedule= lr_schedual),
             stModelCheckpoint("./tmp/RMBdt_weights_{epoch:02d}_loss_{loss:.3f}_valloss_{val_loss:.3f}.h5"),
             ]
model.fit_generator(train_gen,steps_per_epoch=len(train_gen),epochs=epochs,callbacks=callbacks,
                    validation_data=val_gen,validation_steps=len(val_gen))

ring()