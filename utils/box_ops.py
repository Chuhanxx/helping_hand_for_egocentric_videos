# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union+0.0001)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)



    
def load_hand_boxes(hand_info, ind, box_type = 'hand_dets'):
    ind = ind % 600
    max_boxes = 2
    out_boxes = torch.zeros(max_boxes,4)
    if int(ind) in hand_info:
        dets = hand_info[int(ind)][box_type]
        if dets is not None:
            boxes = torch.tensor(dets[:,:4])
            scores = torch.tensor(dets[:,4:5])[:,0]
            topk = torch.argsort(scores,descending=True)[:max_boxes]
            out_boxes[:len(topk)] = boxes[topk]
    return out_boxes


def crop_boxes(boxes, crop_params, ori_im_sz=None, resize_target=None):
    """
    Crop the boxes given the crop parameters.
    Args:
        boxes (array): boxes to crop, in pixels
        crop parameters (int): [y1,x1,h,w]
    """
    boxes_ = boxes.clone()

    if crop_params.sum() <1:
        if ori_im_sz is not None:
            boxes[..., [0, 2]] = boxes[..., [0, 2]] * resize_target /ori_im_sz[1]
            boxes[..., [1, 3]] = boxes[..., [1, 3]] * resize_target /ori_im_sz[0]
        # all zero crop_params = no cropping is needed 
        return boxes
    y1_offset,x1_offset = crop_params[:2]
    x2_max = crop_params[0] + crop_params[3]
    y2_max = crop_params[1] + crop_params[2]

    boxes_[..., [0, 2]] = boxes_[..., [0, 2]] - x1_offset
    boxes_[..., [1, 3]] = boxes_[..., [1, 3]] - y1_offset

    boxes_[..., [0, 2]] = torch.clamp(boxes_[...,[0, 2]], min=0, max=x2_max)
    boxes_[..., [1, 3]] = torch.clamp(boxes_[...,[1, 3]], min=0, max=y2_max)

    if resize_target is not None:
        boxes_[..., [0, 2]] = boxes_[..., [0, 2]] * resize_target /crop_params[-1]
        boxes_[..., [1, 3]] = boxes_[..., [1, 3]] * resize_target /crop_params[-2]
    return boxes_