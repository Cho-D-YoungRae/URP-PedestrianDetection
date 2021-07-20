import torch
import torchvision.transforms.functional as TF


def resize(image, boxes, dims=(300, 300), return_percent_coords=True):
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

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


def transform(image, bboxes, category_ids, is_crowds, img_type, split):

    assert split in {'train', 'test'}

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if img_type == 'lwir':
        mean = sum(mean) / len(mean)
        std = sum(std) / len(std)

    new_image = image
    new_boxes = bboxes
    new_labels = category_ids
    new_difficulties = is_crowds

    new_image, new_boxes = resize(new_image, new_boxes, dims=(300, 300))
    new_image = TF.to_tensor(new_image)
    new_image = TF.normalize(new_image, mean=mean, std=std)

    return new_image, new_boxes, new_labels, new_difficulties
