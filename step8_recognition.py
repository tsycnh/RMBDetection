from keras.models import load_model
from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout,BatchNormalization,Permute,Bidirectional,LSTM,TimeDistributed
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras.utils import Sequence
from rmbgenerator import RMBGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN,LearningRateScheduler
import time
from keras.optimizers import rmsprop,adam,sgd
from stloss import stloss,stModelCheckpoint
from utils import ring

# 参考：https://github.com/xiaofengShi/CHINESE-OCR/blob/master/train/keras-train/model.py
model = Sequential([
    Conv2D(filters=64,kernel_size=(3,3),padding="same",input_shape=(32,128,1)),
    MaxPooling2D(),
    Conv2D(128,(3,3),padding='same'),
    MaxPooling2D(),
    Conv2D(256,(3,3),padding='same'),
    Conv2D(256,(3,3),padding='same'),
    MaxPooling2D((2,1)),
    Conv2D(512,(3,3),padding="same"),
    BatchNormalization(),
    Conv2D(512,(3,3),padding="same"),
    BatchNormalization(),
    MaxPooling2D((2, 1)),
    Conv2D(512,(2,2)),
    Permute(dims=(2,1,3)),
    TimeDistributed(Flatten()),
    Bidirectional(LSTM(256,return_sequences=True)),
    Bidirectional(LSTM(256,return_sequences=True)),
    # Bidirectional(LSTM(256)),
])
model.summary()