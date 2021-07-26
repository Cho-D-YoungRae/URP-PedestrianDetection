from typing import Optional
import torch
import torchvision.transforms.functional as TF
from utils import *
from PIL import Image
from tqdm.auto import tqdm
import json
import Evaluation_official
import os.path
import dataset


def get_object_list(model,
                    original_image,
                    min_score,
                    max_overlap,
                    top_k,
                    ch_option,
                    suppress=None):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Transform
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if ch_option.get('num_ch') == 1:
        if ch_option.get('one_ch_option') == 'mean':
            mean = sum(mean) / len(mean)
            std = sum(std) / len(std)
    image = TF.normalize(TF.to_tensor(TF.resize(original_image, size=(300, 300))),
                         mean=mean, std=std)

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')
    det_labels = det_labels[0].cpu()
    det_scores = det_scores[0].cpu()

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    det_boxes = det_boxes.tolist()
    det_labels = det_labels.tolist()
    det_scores = det_scores.tolist()
    
    return det_boxes, det_labels, det_scores


def get_image(data_dir, img_path, ch_option):
    img_dir, img_name = os.path.split(img_path)
    if ch_option.get('num_ch') == 1:
        if ch_option.get('img_type') in {'lwir', 'visible'}:
            img_path = os.path.join(
                img_dir, ch_option.get('img_type'), img_name)
            image = Image.open(img_path)
            image = image.convert('L')

    elif ch_option.get('num_ch') == 3:
        if ch_option.get('img_type') in {'lwir', 'visible'}:
            img_path = os.path.join(
                data_dir, img_dir, ch_option.get('img_type'), img_name)
            image = Image.open(img_path)
            image = image.convert('RGB')
        elif ch_option.get('img_type') == 'add':
            c_img_path = os.path.join(
                data_dir, img_dir, 'visible', img_name)
            c_image = Image.open(c_img_path)
            
            t_img_path = os.path.join(
                data_dir, img_dir, 'lwir', img_name)
            t_image = Image.open(t_img_path)
            
            image = (TF.to_tensor(c_image) + TF.to_tensor(t_image)) / 2
            image = TF.to_pil_image(image)
            
    return image


def evaluate(model,
             min_score=0.2,
             max_overlap=0.5,
             top_k=200,
             ch_option=dataset.default_ch_option,
             json_path='submission.json',
             data_dir="/content/drive/MyDrive/2021.summer_URP/PD/KAIST_PD"):

    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    data_name_txt = 'test-all-20.txt'
    with open(os.path.join(data_dir, data_name_txt), 'r') as f:
        data_paths = f.readlines()
    data_paths = list(map(lambda x: x[:-1], data_paths))

    img_paths = list(map(
        lambda x: os.path.join('images', x+'.jpg'),
        data_paths
    ))

    detections = []
    
    for image_id, img_path in tqdm(enumerate(img_paths, 0)):
        image = get_image(data_dir, img_path, ch_option)

        det_boxes, det_labels, det_scores = get_object_list(model=model,
                                                            original_image=image,
                                                            min_score=min_score,
                                                            max_overlap=max_overlap,
                                                            top_k=top_k,
                                                            one_ch_option=ch_option)
        
        for i in range(len(det_labels)):
            det_label = det_labels[i]
            det_box = det_boxes[i]
            det_score = det_scores[i]
            x_min, y_min, x_max, y_max = det_box
            w, h = (x_max - x_min) , (y_max - y_min)
            det_box = [x_min, y_min, w, h]
            detection = {
                "image_id": image_id,
                "category_id": det_label,
                "bbox": det_box,
                "score": det_score
            }
            detections.append(detection)
            
    json_path = os.path.join(json_path)
    with open(json_path, 'w', encoding='utf-8') as j:
        json.dump(detections, j, indent='\t')
        
    Evaluation_official.evaluate_coco(json_path)
