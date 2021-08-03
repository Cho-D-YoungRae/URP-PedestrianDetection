from torch.utils.data import Dataset
import torch
import json
import os
from PIL import Image
from transforms import default_transform
import torchvision.transforms.functional as TF

"""
ch_options
num_ch: 1, img_type: lwir, one_ch_option: mean
num_ch: 3, img_type: visible
num_ch: 3, img_type: add

ch_options_X
num_ch: 4, img_type: all, one_ch_option: mean
"""

default_ch_option = {'num_ch': 1,
                     'img_type': 'lwir',
                     'one_ch_option': 'mean'}


class KaistPDDataset(Dataset):

    def __init__(self, 
                 data_dir="/content/drive/MyDrive/2021.summer_URP/PD/KAIST_PD",
                 ch_option=default_ch_option,
                 transform=default_transform,
                 is_sds: bool=False,
                 split: str='train',
                 keep_strange: bool=False,
                 ):
        self.split = split.lower()
        self.data_dir = data_dir
        self.keep_strange = keep_strange
        self.ch_option = ch_option
        self.transform = transform
        self.is_sds = is_sds

        assert self.split in {'train', 'test'}

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
        image = self._get_image(idx)

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
            
        data = []
        if self.is_sds:
            seg_gt_size = (image.height, image.width)
            pseudo_seg_gt = torch.zeros(seg_gt_size)
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                pseudo_seg_gt[y_min:y_max+1, x_min:x_max+1] = 1
            data.append(pseudo_seg_gt.unsqueeze(0))
            
        bboxes = torch.FloatTensor(bboxes)
        category_ids = torch.LongTensor(category_ids)
        is_crowds = torch.ByteTensor(is_crowds)
        
        if self.transform:
            image, bboxes, category_ids, is_crowds = self.transform(image, 
                                                                    bboxes=bboxes,
                                                                    category_ids=category_ids,
                                                                    is_crowds=is_crowds,
                                                                    ch_option=self.ch_option)                
        data = [image, bboxes, category_ids, is_crowds] + data
        
        return data
    
    
    def _get_image(self, idx):
        img_dir, img_name = os.path.split(self.img_paths[idx])
        if self.ch_option.get('num_ch') == 1:
            if self.ch_option.get('img_type') in {'lwir', 'visible'}:
                img_path = os.path.join(
                    self.data_dir, img_dir, self.ch_option.get('img_type'), img_name)
                image = Image.open(img_path)
                image = image.convert('L')

        elif self.ch_option.get('num_ch') == 3:
            if self.ch_option.get('img_type') in {'lwir', 'visible'}:
                img_path = os.path.join(
                    self.data_dir, img_dir, self.ch_option.get('img_type'), img_name)
                image = Image.open(img_path)
                image = image.convert('RGB')
            elif self.ch_option.get('img_type') == 'add':
                c_img_path = os.path.join(
                    self.data_dir, img_dir, 'visible', img_name)
                c_image = Image.open(c_img_path)
                
                t_img_path = os.path.join(
                    self.data_dir, img_dir, 'lwir', img_name)
                t_image = Image.open(t_img_path)
                
                image = (TF.to_tensor(c_image) + TF.to_tensor(t_image)) / 2
                image = TF.to_pil_image(image)
                
        return image
    
    
    
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

def sds_collate_fn(batch):
    images = []
    bboxes = []
    category_ids = []
    is_crowds = []
    pseudo_seg_gts = []

    for b in batch:
        images.append(b[0])
        bboxes.append(b[1])
        category_ids.append(b[2])
        is_crowds.append(b[3])
        pseudo_seg_gts.append(b[4])
        
    images = torch.stack(images, dim=0)
    pseudo_seg_gts = torch.stack(pseudo_seg_gts, dim=0)
    return images, bboxes, category_ids, is_crowds, pseudo_seg_gts
