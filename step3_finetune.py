from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential,load_model
from keras.layers import Dense,Flatten,Dropout,Conv2D,GlobalAveragePooling2D
from keras import losses
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import TensorBoard,ModelCheckpoint,TerminateOnNaN
from keras.optimizers import rmsprop
import time,os

model_path = "C:\\All\\Tdevelop\\RMBDetection\\weights\\step_2-16.h5"
learning_rate = 1e-5
model = load_model(model_path)
# plot_model(model.layers[0])# 绘制主干模型
trainable = False
for layer in model.layers[0].layers:
    if trainable:
        layer.trainable = True
    else:
        layer.trainable = False
    if layer.name == "mixed9":
        trainable = True
model.compile(optimizer=rmsprop(lr=learning_rate),loss=losses.categorical_crossentropy,metrics=['acc'])

model.summary()
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

callbacks = [TensorBoard("./logs/"+ format_time,write_graph=False),
             TerminateOnNaN(),
             ModelCheckpoint("./tmp/RMB_weights_{epoch:02d}_valloss_{val_loss:.2f}_valacc_{val_acc:.2f}.h5")]
history = model.fit_generator(train_datagen,
                    steps_per_epoch=len(train_datagen),
                    epochs=20,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=val_datagen,
                    validation_steps=len(val_datagen))
