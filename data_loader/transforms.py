from torchvision import transforms
from torchvision.transforms._transforms_video import RandomCropVideo, RandomResizedCropVideo,CenterCropVideo, NormalizeVideo,ToTensorVideo,RandomHorizontalFlipVideo
from utils.box_ops import box_cxcywh_to_xyxy
import torch
import  torchvision.transforms.functional as F
from utils.utils import img_denorm
import torchvision
import random 

def init_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225)):
    normalize = transforms.Normalize(mean=norm_mean, std=norm_std)
    tsfm_dict = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(input_res, scale=randcrop_scale),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ]),
        'test': transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,
        ])
    }
    return tsfm_dict

def init_video_transform_dict(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225),
                        force_centercrop=False,
                        resize_wo_crop = True):
    print('Video Transform is used!')
    normalize = NormalizeVideo(mean=norm_mean, std=norm_std)
    if resize_wo_crop:
        val_transform = transforms.Compose([
            transforms.Resize((input_res,input_res)),
            normalize,])
    else:
        val_transform =  transforms.Compose([
            transforms.Resize(center_crop),
            transforms.CenterCrop(center_crop),
            transforms.Resize(input_res),
            normalize,])

    tsfm_dict = {
        'train': transforms.Compose([
            RandomResizedCropVideo(input_res, scale=randcrop_scale),
            RandomHorizontalFlipVideo(),
            transforms.ColorJitter(brightness=color_jitter[0], saturation=color_jitter[1], hue=color_jitter[2]),
            normalize,
        ]),
        'val': val_transform,
        'test': val_transform,
    }
    if force_centercrop:
        tsfm_dict['train'] = tsfm_dict['val']      
    return tsfm_dict


def custom_img_crop(imgs, boxes, pred=False):
    """_summary_

    Args:
        imgs (Tensor): images of a clip [n_frame,3,height,width]
        boxes (Tensor): hand object boxes [n_frame, n_noxes, 4]

    Returns:
        Tensor: cropped images [y1,x1,h,w]
    """
    T = boxes.shape[0]
    im_sz = torch.tensor(imgs.shape[2:])
    

    if pred:
        boxes_area =  boxes[...,2]*im_sz[1] * boxes[...,3]*im_sz[0]

        boxes = box_cxcywh_to_xyxy(boxes)
        boxes[...,0::2] = boxes[...,0::2]*im_sz[1]
        boxes[...,1::2] = boxes[...,1::2]*im_sz[0]

        for j in range(6):
            try:
                biggest = torch.argsort(boxes_area,-1,descending=True)[...,j]
                biggest = biggest[:,None,None].repeat(1,1,4)
                boxes = boxes.scatter(-2, biggest, value=0)
            except:
                import ipdb; ipdb.set_trace()
        boxes = boxes.flatten(0,1)
        boxes = boxes[boxes.sum(-1)!=0]
    else:
        ori_boxes = boxes.clone()
        boxes = boxes[boxes.sum(-1)!=0]
        # find the union of boxes 
    x1, y1 = (torch.min(boxes[...,0],-1)[0], torch.min(boxes[...,1],-1)[0])
    x2, y2 = (torch.max(boxes[...,2],-1)[0], torch.max(boxes[...,3],-1)[0])

    # merge the box union over all the frames, by taking median 
    m_x1 = x1.to(torch.int)
    m_y1 = y1.to(torch.int)
    m_x2 = (torch.max(m_x1, x2)).to(torch.int)
    m_y2 = (torch.max(m_y1, y2)).to(torch.int)
    m_cx, m_cy = ((m_x1+m_x2)/2).to(torch.int),((m_y1+m_y2)/2).to(torch.int)
    w_, h_= (m_x2-m_x1, m_y2-m_y1)

    if w_ < 5 or h_< 5:
        return imgs, torch.tensor([0.,0.,0.,0.])
    long_side = max(h_, w_)

    attempt = 0
    # while   h_*w_< im_sz[0]*im_sz[1]*(random.random()/2):
    while  h_*w_ < im_sz[0]*im_sz[1]*0.5 and attempt < 100:
        w_ = (w_ * 1.2).to(torch.int)
        h_ = (h_ * 1.2).to(torch.int)
        long_side = max(h_, w_)
        attempt+= 1
    newx1 = torch.max(torch.zeros(m_cx.shape),m_cx-w_/2).to(torch.int)
    newy1 = torch.max(torch.zeros(m_cy.shape),m_cy-h_/2).to(torch.int)

    if torch.min(im_sz[0]-newy1,long_side) <1 or torch.min(im_sz[1]-newx1,long_side) < 1:
        return imgs, torch.tensor([0,0,0,0])
    new_imgs = F.crop(imgs, newy1, newx1, torch.min(im_sz[0]-newy1,long_side), torch.min(im_sz[1]-newx1,long_side))
    crop_params = torch.stack([newy1, newx1, torch.min(im_sz[0]-newy1,long_side), torch.min(im_sz[1]-newx1,long_side)])
    
    # from utils.utils import draw_bbox
    # from PIL import Image
    # vis_img = draw_bbox(imgs[0], ori_boxes[0])
    # vis_img.save('vis.png')
    # torchvision.utils.save_image(imgs,'ori_img_dataloader.png')
    # torchvision.utils.save_image(new_imgs,'cropped_img_dataloader.png')
    # # import ipdb; ipdb.set_trace()
    # import ipdb; ipdb.set_trace()

    return new_imgs, crop_params


