from keras.models import Sequential,load_model
from keras import losses
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN
from keras.optimizers import rmsprop
from keras.utils import plot_model
import time,os
from rmbgenerator import RMBGenerator
from stloss import stloss,stModelCheckpoint

model = load_model("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_5-10.h5")

learning_rate = 1e-4
epochs = 300
plot_model(model.layers[0])# 绘制主干模型
trainable = False
for layer in model.layers[0].layers:
    if trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    if layer.name == "activation_46":
        trainable = True
model.compile(optimizer=rmsprop(lr=learning_rate),loss=stloss)
model.summary()

# 2. prepare data
# 准备数据
train_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\train\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\train\\annos",
                         batch_size=32,rescale=1.0/255)
val_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\val\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\val\\annos",
                         batch_size=4,rescale=1.0/255)
format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())


callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             stModelCheckpoint("./tmp/RMBdt_weights_{epoch:02d}_loss_{loss:.2f}_valloss_{val_loss:.2f}.h5"),
             ]
model.fit_generator(train_gen,steps_per_epoch=len(train_gen),epochs=epochs,callbacks=callbacks,
                    validation_data=val_gen,validation_steps=len(val_gen))

import ctypes
import time
player = ctypes.windll.kernel32

for i in range(10):
    time.sleep(1)
    player.Beep(1000,200)