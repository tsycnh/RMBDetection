## 人民币编码识别

### 1.赛题说明
由Tinymind主办的人民币编码识别大赛，分别包括面值识别和冠字号码文字识别两大部分组成。

比赛官网：https://www.tinymind.cn/competitions/47  
比赛时间：2019年5月9日 ~ 2019年12月31日

数据集:   
链接: https://pan.baidu.com/s/1nt03m3JddC93HIx0II-z5w 提取码: nk8g  
数据包含训练集和测试集两大部分  
```
├─训练集
│       train_data.z01
│       train_data.z02
│       train_data.z03
│       train_data.z04
│       train_data.z05
│       train_data.zip
│       train_face_value_label.csv
│       train_id_label.csv
└─测试集
       private_test_data.zip
       public_test_data.z01
       public_test_data.zip
```
其中 train_data.* 为图像数据，共39620张图像，如下图  
![五元](./resource/fiveyuan.jpg)  

train_face_value_label.csv 为面值大小标记文件，以文件名和面值大小一一对应：
```buildoutcfg
013MNV9B.jpg, 100
016ETNGG.jpg, 50
018SUTBA.jpg, 0.1
0192G5IC.jpg, 5
```

train_id_label.csv 为冠字号码标记文件：
```buildoutcfg
GK5NXT2E.jpg, SJ88154371
PNFRISAL.jpg, XI90599371
2GVHF7RK.jpg, WG35669371
DGUVOETG.jpg, YH01075371
```
（注：根据一些比赛队伍表示，标记中存在个别错误的label）  

训练集中没有对冠字号码的ROI区域进行标记，可能需要自己手动标记。

### 2. 解决方案
### 2.1 面值识别

面值识别本质就是最简单的分类问题，考虑采用keras加载预训练模型实现。

<code>step1_split_train_dataset.py</code>：整理数据集  
训练集用20000张，其余的19620张作为验证集