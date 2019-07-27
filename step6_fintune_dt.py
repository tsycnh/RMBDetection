from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential,load_model
from keras.layers import Dense,Flatten,Dropout,Conv2D,GlobalAveragePooling2D
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN
from keras.optimizers import rmsprop
from keras.utils import plot_model
import time,os
from rmbgenerator import RMBGenerator

model = load_model("C:\\All\\Tdevelop\\RMBDetection\\tmp\\RMBdt_weights_19_valloss_0.56.h5")
# model = load_model("C:\\All\\Tdevelop\\RMBDetection\\tmp\\RMBdt_weights_17_valloss_4.17.h5")

learning_rate = 1e-6
plot_model(model.layers[0])# 绘制主干模型
trainable = False
for layer in model.layers[0].layers:
    if trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    if layer.name == "activation_46":
        trainable = True
model.compile(optimizer=rmsprop(lr=learning_rate),loss=losses.binary_crossentropy)
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
             ModelCheckpoint("./tmp/RMBdt_weights_{epoch:02d}_valloss_{val_loss:.2f}.h5"),
             ]
model.fit_generator(train_gen,steps_per_epoch=len(train_gen),epochs=10,callbacks=callbacks,
                    validation_data=val_gen,validation_steps=len(val_gen))