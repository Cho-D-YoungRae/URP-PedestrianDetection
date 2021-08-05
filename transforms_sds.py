import torch
import random
import torchvision.transforms.functional as TF
from utils import *
import transforms
from torch.nn.functional import interpolate



def expand(image, boxes, pseudo_seg_gt, filler):
    """
    Perform a zooming out operation by placing the image in a larger canvas of filler material.

    Helps to learn to detect smaller objects.

    :param image: image, a tensor of dimensions (3, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param filler: RBG values of the filler material, a list like [R, G, B]
    :return: expanded image, updated bounding box coordinates
    """
    # Calculate dimensions of proposed expanded (zoomed-out) image
    original_h = image.size(1)
    original_w = image.size(2)
    max_scale = 4
    scale = random.uniform(1, max_scale)
    new_h = int(scale * original_h)
    new_w = int(scale * original_w)

    # Create such an image with the filler
    filler = torch.FloatTensor(filler)  # (3)
    num_channel = image.size(0)
    new_image = torch.ones((num_channel, new_h, new_w), dtype=torch.float) * filler.unsqueeze(1).unsqueeze(1)  # (3, new_h, new_w)
    # Create such a segmentation gt with 0
    new_seg_gt = torch.zeros(1, new_h, new_w)   ###
    # Note - do not use expand() like new_image = filler.unsqueeze(1).unsqueeze(1).expand(3, new_h, new_w)
    # because all expanded values will share the same memory, so changing one pixel will change all

    # Place the original image at random coordinates in this new image (origin at top-left of image)
    left = random.randint(0, new_w - original_w)
    right = left + original_w
    top = random.randint(0, new_h - original_h)
    bottom = top + original_h
    new_image[:, top:bottom, left:right] = image
    # Place the original pseudo segmentation gt at random coordinates in this new segmentation gt
    new_seg_gt[:, top:bottom, left:right] = pseudo_seg_gt   ###

    # Adjust bounding boxes' coordinates accordingly
    new_boxes = boxes + torch.FloatTensor([left, top, left, top]).unsqueeze(
        0)  # (n_objects, 4), n_objects is the no. of objects in this image

    return new_image, new_boxes, new_seg_gt


def random_crop(image, boxes, labels, difficulties, pseudo_seg_gt):
    """
    Performs a random crop in the manner stated in the paper. Helps to learn to detect larger and partial objects.

    Note that some objects may be cut out entirely.

    Adapted from https://github.com/amdegroot/ssd.pytorch/blob/master/utils/augmentations.py

    :param image: image, a tensor of dimensions (nun_ch, original_h, original_w)
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :param labels: labels of objects, a tensor of dimensions (n_objects)
    :param difficulties: difficulties of detection of these objects, a tensor of dimensions (n_objects)
    :return: cropped image, updated bounding box coordinates, updated labels, updated difficulties
    """
    original_h = image.size(1)
    original_w = image.size(2)
    # Keep choosing a minimum overlap until a successful crop is made
    while True:
        # Randomly draw the value for minimum overlap
        min_overlap = random.choice([0., .1, .3, .5, .7, .9, None])  # 'None' refers to no cropping

        # If not cropping
        if min_overlap is None:
            return image, boxes, labels, difficulties, pseudo_seg_gt

        # Try up to 50 times for this choice of minimum overlap
        # This isn't mentioned in the paper, of course, but 50 is chosen in paper authors' original Caffe repo
        max_trials = 50
        for _ in range(max_trials):
            # Crop dimensions must be in [0.3, 1] of original dimensions
            # Note - it's [0.1, 1] in the paper, but actually [0.3, 1] in the authors' repo
            min_scale = 0.3
            scale_h = random.uniform(min_scale, 1)
            scale_w = random.uniform(min_scale, 1)
            new_h = int(scale_h * original_h)
            new_w = int(scale_w * original_w)

            # Aspect ratio has to be in [0.5, 2]
            aspect_ratio = new_h / new_w
            if not 0.5 < aspect_ratio < 2:
                continue

            # Crop coordinates (origin at top-left of image)
            left = random.randint(0, original_w - new_w)
            right = left + new_w
            top = random.randint(0, original_h - new_h)
            bottom = top + new_h
            crop = torch.FloatTensor([left, top, right, bottom])  # (4)

            # Calculate Jaccard overlap between the crop and the bounding boxes
            overlap = find_jaccard_overlap(crop.unsqueeze(0),
                                           boxes)  # (1, n_objects), n_objects is the no. of objects in this image
            overlap = overlap.squeeze(0)  # (n_objects)

            # If not a single bounding box has a Jaccard overlap of greater than the minimum, try again
            if overlap.max().item() < min_overlap:
                continue

            # Crop image
            new_image = image[:, top:bottom, left:right]  # (3, new_h, new_w)
            # Crop pseudo segmentation gt
            new_seg_gt = pseudo_seg_gt[:, top:bottom, left:right]   ###

            # Find centers of original bounding boxes
            bb_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.  # (n_objects, 2)

            # Find bounding boxes whose centers are in the crop
            centers_in_crop = (bb_centers[:, 0] > left) * (bb_centers[:, 0] < right) * (bb_centers[:, 1] > top) * (
                    bb_centers[:, 1] < bottom)  # (n_objects), a Torch uInt8/Byte tensor, can be used as a boolean index

            # If not a single bounding box has its center in the crop, try again
            if not centers_in_crop.any():
                continue

            # Discard bounding boxes that don't meet this criterion
            new_boxes = boxes[centers_in_crop, :]
            new_labels = labels[centers_in_crop]
            new_difficulties = difficulties[centers_in_crop]

            # Calculate bounding boxes' new coordinates in the crop
            new_boxes[:, :2] = torch.max(new_boxes[:, :2], crop[:2])  # crop[:2] is [left, top]
            new_boxes[:, :2] -= crop[:2]
            new_boxes[:, 2:] = torch.min(new_boxes[:, 2:], crop[2:])  # crop[2:] is [right, bottom]
            new_boxes[:, 2:] -= crop[:2]

            return new_image, new_boxes, new_labels, new_difficulties, new_seg_gt


def flip(image, boxes, pseudo_seg_gt):
    """
    Flip image horizontally.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: flipped image, updated bounding box coordinates
    """
    # Flip image
    new_image = TF.hflip(image)

    # Flip boxes
    new_boxes = boxes
    new_boxes[:, 0] = image.width - boxes[:, 0] - 1
    new_boxes[:, 2] = image.width - boxes[:, 2] - 1
    new_boxes = new_boxes[:, [2, 1, 0, 3]]
    
    # Flip pseudo segmentation gt
    new_seg_gt = torch.flip(pseudo_seg_gt, dims=(0, 2))

    return new_image, new_boxes, new_seg_gt


def resize(image, boxes, pseudo_seg_gt, dims=(300, 300), return_percent_coords=True):
    """
    Resize image. For the SSD300, resize to (300, 300).

    Since percent/fractional coordinates are calculated for the bounding boxes (w.r.t image dimensions) in this process,
    you may choose to retain them.

    :param image: image, a PIL Image
    :param boxes: bounding boxes in boundary coordinates, a tensor of dimensions (n_objects, 4)
    :return: resized image, updated bounding box coordinates (or fractional coordinates, in which case they remain the same)
    """
    # Resize image
    new_image = TF.resize(image, dims)

    # Resize pseudo segmentation gt
    new_seg_gt = interpolate(pseudo_seg_gt.unsqueeze(0), size=dims, mode='nearest')
    new_seg_gt = new_seg_gt.squeeze(0)
    
    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes, new_seg_gt


def default_transform(image, bboxes, category_ids, is_crowds, pseudo_seg_gt, ch_option):
    
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if ch_option.get('num_ch') == 1:
        if ch_option.get('one_ch_option') == 'mean':
            mean = [sum(mean) / len(mean)]
            std = [sum(std) / len(std)]

    new_image = image
    new_bboxes = bboxes
    new_category_ids = category_ids
    new_is_crowds = is_crowds
    new_pseudo_seg_gt = pseudo_seg_gt
    # Skip the following operations for evaluation/testing
    
    # A series of photometric distortions in random order, each with 50% chance of occurrence, as in Caffe repo
    new_image = transforms.photometric_distort(new_image)

    # Convert PIL image to Torch tensor
    new_image = TF.to_tensor(new_image)
    new_image = TF.normalize(new_image, mean=mean, std=std)

    # Expand image (zoom out) with a 50% chance - helpful for training detection of small objects
    # Fill surrounding space with the mean of ImageNet data that our base VGG was trained on
    if random.random() < 0.5:
        new_image, new_bboxes, new_pseudo_seg_gt =\
            expand(new_image, bboxes, new_pseudo_seg_gt, filler=mean)

    # Randomly crop image (zoom in)
    new_image, new_bboxes, new_category_ids, new_is_crowds, new_pseudo_seg_gt = \
        random_crop(new_image, new_bboxes, new_category_ids, new_is_crowds, new_pseudo_seg_gt)

    # Convert Torch tensor to PIL image
    new_image = TF.to_pil_image(new_image)

    # Flip image with a 50% chance
    if random.random() < 0.5:
        new_image, new_bboxes, new_pseudo_seg_gt =\
            flip(new_image, new_bboxes, new_pseudo_seg_gt)

    # Resize image to (300, 300) - this also converts absolute boundary coordinates to their fractional form


    # Convert PIL image to Torch tensor
    new_image = TF.to_tensor(new_image)

    # Normalize by mean and standard deviation of ImageNet data that our base VGG was trained on
    # new_image = TF.normalize(new_image, mean=mean, std=std)

    return new_image, new_bboxes, new_category_ids, new_is_crowds, new_pseudo_seg_gt
