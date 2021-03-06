from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense,Flatten,Dropout,Conv2D,GlobalAveragePooling2D
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN
from keras.optimizers import rmsprop
import time,os
# 1. construct model
img_shape = (400,400,3)
learning_rate = 1e-4
load_weights = False

backbone = InceptionV3(include_top=False,weights='imagenet',input_shape=img_shape)
model = Sequential([
    backbone,
    # Conv2D(filters=256,kernel_size=(3,3)),
    # GlobalAveragePooling2D(),
    Flatten(),
    Dense(256,activation='relu',),
    Dropout(0.5),
    Dense(128,activation='relu',),
    Dropout(0.25),
    Dense(9,activation="softmax")
])
if load_weights:
    model.load_weights("C:\\All\\Tdevelop\\RMBDetection\\weights\\step_2-13.h5")
backbone.trainable = False
model.summary()

model.compile(optimizer=rmsprop(lr=learning_rate),loss=losses.categorical_crossentropy,metrics=['acc'])

# 2. prepare data
data_dir = "C:\\All\\Data\\RMB\\NEW_MINI"
train_dir = os.path.join(data_dir,'Train')
val_dir = os.path.join(data_dir,'Val')
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1
                                   ).flow_from_directory(train_dir,target_size=(400,400),batch_size=32)
val_datagen= ImageDataGenerator(rescale=1./255).flow_from_directory(val_dir,target_size=(400,400),batch_size=32)
format_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
print("class indices: ",train_datagen.class_indices)
callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             ModelCheckpoint("./tmp/RMB_weights_{epoch:02d}_valloss_{val_loss:.2f}_valacc_{val_acc:.2f}.h5")]
history = model.fit_generator(train_datagen,
                    steps_per_epoch=len(train_datagen),
                    epochs=20*5,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_datagen,
                    validation_steps=len(val_datagen))
