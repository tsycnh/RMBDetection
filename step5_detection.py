from keras.models import load_model
from keras.models import Model,Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dense
from keras.applications.resnet50 import ResNet50
from keras.utils import Sequence
'''
# 方案1
# 以yolov3-tiny为基础打造。
backbone = load_model("weights/yolov3-tiny.h5")
backbone.summary(positions=[.2, .55, .67, 1.])

output1,output2=backbone.outputs
o1 = Conv2D(15,kernel_size=(1,1),name='conv2d_append1')(output1)
o2 = Conv2D(15,kernel_size=(1,1),name='conv2d_append2')(output2)

model = Model(inputs=backbone.input,outputs=(o1,o2))

model.summary(positions=[.2, .55, .67, 1.])

anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
'''
backbone = ResNet50(include_top=False,weights='imagenet',input_shape=(400,400,3))
backbone.summary(positions=[.22, .55, .67, 1.])

model = Sequential([
    backbone,
    MaxPooling2D(),
    Conv2D(512,kernel_size=(1,1),padding="same",activation='relu'),
    Conv2D(512,kernel_size=(3,3),activation='relu'),
    Flatten(),
    Dense(1024,activation='relu'),
    Dense(4,activation='sigmoid')
])
model.summary(positions=[.22, .55, .67, 1.])
# 输出的四个节点分别表示center_x,center_y,w,h。这四个值都是bbox相对于图像长宽的值，取值范围均为0~1

model.compile(optimizer='rmsprop',loss='binary_crossentropy')

model.fit_generator()

