import os
import cv2
import numpy as np
import torch.utils.data as data

class CCDataset(data.Dataset):
    def __init__(self,
                 is_train :bool = False,
                 input_width : int = 168,
                 input_height : int = 48,
                 data_dir :str = None,
                ) -> None:
        super().__init__()

        self.is_train = is_train
        self.data_dir = data_dir
        self.input_width = input_width
        self.input_height = input_height
        
        self.mean = np.array(0.588, dtype=np.float32)
        self.std = np.array(0.193, dtype=np.float32)
        
        text_file = 'train.txt' if is_train else 'val.txt'
        
        self.labels = []
        for data_dir in self.data_dir:
            with open(os.path.join(data_dir, text_file), 'r', encoding='utf-8') as f:
                self.labels.extend([{os.path.join(data_dir, 'JPEGImages', c.split('\t')[0]): c.split('\t')[-1][:-1]} for c in f.readlines()])
        print('==============================')
        print('successed load {} images'.format(self.__len__()))
    
    def __getitem__(self, index):
        image_path = list(self.labels[index].keys())[0]
        
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image.shape[-1]==4:
            image=cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        img_h, img_w, _ = image.shape
        
        image = cv2.resize(image, (self.input_width, self.input_height)).astype(np.float32)
        
        image = (image/255. - self.mean) / self.std
        
        image = image.transpose([2, 0, 1])
            
        return image, list(self.labels[index].values())[0], index
    
    def __len__(self):
        return len(self.labels)
    
    def pull_image(self, index):
        image_path = list(self.labels[index].keys())[0]
        
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        return image

    def pull_anno(self, index):
        return list(self.labels[index].values())[0]