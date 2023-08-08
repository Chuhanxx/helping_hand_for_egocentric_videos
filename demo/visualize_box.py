import torch
import argparse
import os
import decord
import numpy as np
import sys
from PIL import Image

sys.path.insert(0, "../")
from data_loader.transforms import init_video_transform_dict
from utils.utils import  draw_box_on_clip
from utils.box_ops import box_cxcywh_to_xyxy


def main(args):
    demo_colors =  ['blue','green','orange','brown','pink','white','Cyan','gold','Khaki','Indigo','LightBlue','LightSalmon','SlateGray','Chocolate','DarkBlue','DarkGray','DarkSlateGrey','linen','gray','beige']

    tsfm_params = {
        "force_centercrop": True,
        "norm_mean": [108.3272985/255, 116.7460125/255, 104.09373615000001/255],
        "norm_std":[68.5005327/255, 66.6321579/255, 70.32316305/255],
        "normalize":False}
    tsfm_dict = init_video_transform_dict(**tsfm_params)
    tsfm = tsfm_dict['train']

    annotations = torch.load(args.anno_file)
    for anno in annotations:
        chunk_id = str(int(anno['start_sec']//600))
        video_uid = anno['video_uid']
        video_path = os.path.join(args.video_dir, video_uid, chunk_id+'.mp4')
        frames = read_frames(video_path, 
                                   anno['sample_sec'], 
                                   tsfm)
        

        # draw boxes
        vis_imgs = frames.clone()
        color_count = 0
        for k, hand_box in anno['hand_boxes'].items():
            vis_imgs = draw_box_on_clip( box_cxcywh_to_xyxy(hand_box[:,None,:]),
                                        vis_imgs,
                                        word=k,
                                        color=demo_colors[color_count])
            color_count+=1
        for k, obj_box in anno['obj_boxes'].items():
            vis_imgs = draw_box_on_clip( box_cxcywh_to_xyxy(obj_box[:,None,:]),
                                        vis_imgs,
                                        word=k,
                                        color=demo_colors[color_count])
            color_count+=1
        vis_img = np.concatenate(vis_imgs,1)
        vis_img = Image.fromarray(vis_img)
        name = '_'.join(anno['caption'].split())
        vis_img.save(f'grounding_vis/{name}.png')
        import ipdb; ipdb.set_trace()

def read_frames(vpath, frames, transforms):
 
    vr = decord.VideoReader(vpath)
    frame_ids = [int(f*30) for f in frames]
    # load frames
    try:
        frames = vr.get_batch(frame_ids)
    except decord.DECORDError as error:
        print(error)
        frames = vr.get_batch([0] * len(frame_ids))

    frames = frames.to(torch.float32)/255
    frames = frames.permute(3,0,1,2)  # [T, H, W, C] ---> [C, T, H, W]
    frames = transforms(frames)
    frames = frames.transpose(0, 1)  # [C, T, H, W] ---> [T, C, H, W]
    return frames



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', default='/', type=str)
    parser.add_argument('--anno_file', default='', type=str)
    args = parser.parse_args()
    return args        
if __name__ == '__main__':
    args = parse_args()
    main(args)
