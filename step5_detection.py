from keras.models import load_model
from keras.models import Model
from keras.layers import Conv2D
from keras.applications.resnet50 import ResNet50

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