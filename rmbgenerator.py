from keras.utils import Sequence
import os,shutil

# 自定义一个数据生成器。继承自keras.utils.Sequence,可以和fit_generator无缝衔接
# Sequence 是keras中数据生成器的基类。必须实现__getitem__(),__len__(),建议实现on_epoch_end(),可以在这里打乱数据集
class RMBGenerator(Sequence):
    def __init__(self, images_dir, annos_dir, batch_size):
        self.batch_size = batch_size
        self.images_list = os.listdir(images_dir)
        self.annos_list = os.listdir(annos_dir)
        self.images_list.sort()
        self.annos_list.sort()
        self.pair = list(zip(self.images_list,self.annos_list))

        pass
    # def __len__(self):
    #     return int(np.ceil(len(self.x) / float(self.batch_size)))
    #
    # def __getitem__(self, idx):
    #     batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
    #     batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
    #
    #     return np.array([
    #         resize(imread(file_name), (200, 200))
    #         for file_name in batch_x]), np.array(batch_y)

if __name__ == "__main__":
    rmbd = RMBGenerator("C:\\All\\Data\\RMB\\Detection\\train\\images",
                        "C:\\All\\Data\\RMB\\Detection\\train\\annos",
                        32)
    pass