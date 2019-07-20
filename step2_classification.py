from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN
from keras.optimizers import rmsprop
import time
# 1. construct model
backbone = InceptionV3(include_top=False,weights='imagenet',input_shape=(400,400,3))
model = Sequential([
    backbone,
    Flatten(),
    Dense(128,activation='relu',),
    Dropout(0.5),
    Dense(9,activation="softmax")
])
backbone.trainable = False
model.summary()

model.compile(optimizer=rmsprop(lr=1e-4),loss=losses.categorical_crossentropy,metrics=['acc'])

# 2. prepare data
train_dir = "C:\\All\\Data\\RMB\\NEW\\Train"
val_dir = "C:\\All\\Data\\RMB\\NEW\\Val"
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1
                                   ).flow_from_directory(train_dir,target_size=(400,400),batch_size=32)
val_datagen= ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir,target_size=(400,400),batch_size=32)
format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())

callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             ModelCheckpoint("./tmp/RMB_weights_{epoch:02d}_loss_{val_loss:.2f}.h5")]
model.fit_generator(train_datagen,
                    steps_per_epoch=len(train_datagen),epochs=3,callbacks=callbacks,validation_data=val_datagen,validation_steps=len(val_datagen))