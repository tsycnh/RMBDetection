from keras.models import load_model
from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.utils import Sequence
from rmbgenerator import RMBGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN,LearningRateScheduler
import time
from keras.optimizers import rmsprop,adam,sgd
from stloss import stloss,stModelCheckpoint
from utils import ring

backbone = ResNet50(include_top=False,weights='imagenet',input_shape=(400,400,3))
# backbone.summary(positions=[.22, .55, .67, 1.])

model = Sequential([
    backbone,
    Dropout(0.5),
    Conv2D(512,kernel_size=(1,1),padding="same"),
    LeakyReLU(0.1),
    Dropout(0.5),

    Conv2D(1024,kernel_size=(3,3),padding='same'),
    LeakyReLU(0.1),
    MaxPooling2D(),
    Dropout(0.5),

    Conv2D(512, kernel_size=(1, 1), padding="same"),
    LeakyReLU(0.1),
    Dropout(0.5),

    Conv2D(1024, kernel_size=(3, 3), padding='same'),
    LeakyReLU(0.1),
    Dropout(0.5),

    Conv2D(5,(1,1),padding='same'),
    LeakyReLU(0.1),

])
backbone.trainable = False
model.summary(positions=[.22, .55, .67, 1.])
model.compile(optimizer=rmsprop(5e-3),loss=stloss)

# 准备数据
train_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\train\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\train\\annos",
                         batch_size=32,rescale=1.0/255,aug=False)
val_gen = RMBGenerator(images_dir="C:\\All\\Data\\RMB\\Detection\\val\\images",
                         annos_dir="C:\\All\\Data\\RMB\\Detection\\val\\annos",
                         batch_size=4,rescale=1.0/255)
format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

def lr_schedual(epoch,lr):
    if epoch <=3:
        return 1e-3
    else:
        return 1e-4
callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             stModelCheckpoint("./tmp/RMBdt_weights_{epoch:02d}_loss_{loss:.3f}_valloss_{val_loss:.3f}.h5"),
             # LearningRateScheduler(schedule= lr_schedual)
             ]
model.fit_generator(train_gen,steps_per_epoch=len(train_gen),epochs=30,callbacks=callbacks,
                    validation_data=val_gen,validation_steps=len(val_gen))
ring()
