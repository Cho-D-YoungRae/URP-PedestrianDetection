from typing import Optional
from torch.utils.data import Dataset
import torch
import json
import os
from PIL import Image
from transforms import default_transform


class KaistPDDataset(Dataset):

    def __init__(self, 
                 data_dir="/content/drive/MyDrive/2021.summer_URP/PD/KAIST_PD",
                 ch_option: Optional[str]="mean",
                 transform: Optional[function]=default_transform,
                 split: str='train',
                 keep_strange: bool=False,
                 ):
        self.split = split.lower()
        self.data_dir = data_dir
        self.keep_strange = keep_strange
        self.ch_option = ch_option
        self.transform = transform
        self.img_type = 'lwir' if self.ch_option else 'visible'
        self.img_conversion = 'L' if self.img_type == 'lwir' else 'RGB'

        assert self.split in {'train', 'test'}
        assert self.ch_option in {'mean'}

        data_name_txt = self.split + '-all-20.txt'
        with open(os.path.join(data_dir, data_name_txt), 'r') as f:
            data_paths = f.readlines()
        data_paths = list(map(lambda x: x[:-1], data_paths))

        self.anno_paths = list(map(
            lambda x: os.path.join('annotation_json', x+'.json'),
            data_paths
        ))

        self.img_paths = list(map(
            lambda x: os.path.join('images', x+'.jpg'),
            data_paths
        ))

        assert len(self.anno_paths) == len(self.img_paths)

    
    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, idx):
        img_dir, img_name = os.path.split(self.img_paths[idx])
        img_path = os.path.join(
            self.data_dir, img_dir, self.img_type, img_name)
        image = Image.open(img_path)
        image = image.convert(self.img_conversion)
        

        with open(os.path.join(self.data_dir, self.anno_paths[idx]), 'r') as j:
            annotations = json.load(j)['annotation']

        bboxes = []
        category_ids = []
        is_crowds = []
        for annotation in annotations:
            if not self.keep_strange and annotation['category_id'] == -1:
                continue
            bboxes.append(annotation['bbox'])
            category_ids.append(annotation['category_id'])
            is_crowds.append(annotation['is_crowd'])
        bboxes = torch.FloatTensor(bboxes)
        category_ids = torch.LongTensor(category_ids)
        is_crowds = torch.ByteTensor(is_crowds)

        if self.transform:
            image, bboxes, category_ids, is_crowds =\
                self.transform(image, bboxes, category_ids, is_crowds, self.ch_option)

        return image, bboxes, category_ids, is_crowds
    
    
def collate_fn(batch):
    images = []
    bboxes = []
    category_ids = []
    is_crowds = []

    for b in batch:
        images.append(b[0])
        bboxes.append(b[1])
        category_ids.append(b[2])
        is_crowds.append(b[3])

    images = torch.stack(images, dim=0)

    return images, bboxes, category_ids, is_crowds