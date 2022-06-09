
import os
import jittor as jt
from jittor import transform

data_transforms = {
        'train': transform.Compose([
            transform.Resize((256,256)),
            transform.RandomCrop((224, 224)),       # 从中心开始裁剪
            transform.RandomHorizontalFlip(p=0.5),  # 随机水平翻转 选择一个概率
            transform.RandomVerticalFlip(p=0.5),    # 随机垂直翻转
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])  # 均值，标准差
        ]),
        'valid': transform.Compose([
            transform.Resize((224,224)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ]),
        'test': transform.Compose([
            transform.Resize((224,224)),
            transform.ToTensor(),
            transform.ImageNormalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    }

class Dataset():

    def __init__(self, dataset_path="./data", batch_size=16):

        self.data_dir = dataset_path
        image_datasets = {x: jt.dataset.ImageFolder(os.path.join(self.data_dir, x), data_transforms[x]) for x in
                        ['train', 'valid', 'test']}

        self.traindataset = image_datasets['train'].set_attrs(batch_size=batch_size, shuffle=True)
        self.validdataset = image_datasets['valid'].set_attrs(batch_size=64, shuffle=False)
        self.testdataset = image_datasets['test'].set_attrs(batch_size=1, shuffle=False)


        self.dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
        self.train_num = len(self.traindataset)
        self.val_num = len(self.validdataset)
        self.test_num = len(self.testdataset)
