import sys
import torch
import numpy as np
import pickle
import os
import glob
import re
import matplotlib.pyplot as plt
import functools
import json
import socket
import time
import humanize
import psutil
plt.switch_backend('agg')
from datetime import datetime
from collections import deque, OrderedDict
from tqdm import tqdm 
import math
_int_classes = int
import copy 
import argparse
import torchvision
from torchvision import transforms
from PIL import Image, ImageDraw,ImageFont
from itertools import repeat
from pathlib import Path


def denorm(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    assert len(mean)==len(std)==3
    inv_mean = [-mean[i]/std[i] for i in range(3)]
    inv_std = [1/i for i in std]
    return transforms.Normalize(mean=inv_mean, std=inv_std)

def batch_denorm(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], channel=1):
    shape = [1]*tensor.dim(); shape[channel] = 3
    dtype = tensor.dtype 
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device).view(shape)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device).view(shape)
    output = tensor.mul(std).add(mean)
    return output 


def strfdelta(tdelta, fmt):
    d = {"d": tdelta.days}
    d["h"], rem = divmod(tdelta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    return fmt.format(**d)


class Logger(object):
    '''write something to txt file'''
    def __init__(self, path):
        self.birth_time = datetime.now()
        filepath = os.path.join(path, self.birth_time.strftime('%Y-%m-%d-%H:%M:%S')+'.log')
        self.filepath = filepath
        with open(filepath, 'a') as f:
            f.write(self.birth_time.strftime('%Y-%m-%d %H:%M:%S')+'\n')

    def log(self, string):
        with open(self.filepath, 'a') as f:
            time_stamp = datetime.now() - self.birth_time
            f.write(strfdelta(time_stamp,"{d}-{h:02d}:{m:02d}:{s:02d}")+'\t'+string+'\n')


def neq_load_customized(model, pretrained_dict, verbose=True):
    ''' load pre-trained model in a not-equal way,
    when new model has been partially modified '''
    model_dict = model.state_dict()
    tmp = {}
    if verbose:
        print('\n=======Check Weights Loading======')
        print('Weights not used from pretrained file:')
    
    for k, v in pretrained_dict.items():
        if k in model_dict:
            tmp[k] = v
        else:
            if verbose:
                print(k)
    if verbose:
        print('---------------------------')
        print('Weights not loaded into new model:')
        for k, v in model_dict.items():
            if k not in pretrained_dict:
                print(k)
        print('===================================\n')
    
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    del pretrained_dict
    model_dict.update(tmp)
    del tmp
    model.load_state_dict(model_dict)
    return model

def img_denorm(tensor,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                ):    
    

    mean = torch.tensor(mean)[:,None,None].to(tensor.device)
    std = torch.tensor(std)[:,None,None].to(tensor.device)
    tensor = tensor.mul(std).add(mean)
    tensor = tensor.detach()
    img = tensor/tensor.max()
    return img

def draw_bbox(
    tensor,
    bboxes,
    width=2,
    texts=None,
    norm=True,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    color=None
):
    topil = torchvision.transforms.ToPILImage()
    if norm:
        tensor = img_denorm(tensor, mean=mean, std=std)
    if torch.is_tensor(tensor):
        im = topil(tensor)
    else:
        im = tensor
   
    colors = ['blue','green','orange','brown','pink','white','Cyan','gold','Khaki','Indigo','LightBlue','LightSalmon','SlateGray','Chocolate','DarkBlue','DarkGray','DarkSlateGrey','linen','gray','beige']

    for i,bbox in enumerate(bboxes):
        if bbox.sum() == 0 :
            continue
        draw = ImageDraw.Draw(im)
        if color is None:
           color = colors[i]
        draw.rectangle(bbox.tolist(), outline=color, width=width)
        if texts is not None:
            try:
                font = ImageFont.truetype('/users/czhang/EgoVLP/miscs/arial.ttf', size=16)
                draw.text((bbox[0], max(0, bbox[1]-20)), texts[i], font=font, fill=color) 
            except:
                import ipdb; ipdb.set_trace()
                
    return im 

def draw_box_on_clip(bbox, frames, word=None, name='bbox', color=None):
    vis_imgs = []
    for f_i in range(4):
        vis_img = draw_bbox(frames[f_i], 
            bbox[f_i]*224,
            norm =False, 
            width=3, 
            texts=[word],
            color=color)
        vis_imgs.append(vis_img)
    return vis_imgs


def plot_attn_map(attn, n_cols=3, n_rows=2, name =''):
    B,H,W = attn.shape

    fig, axes = plt.subplots(n_rows, n_cols)

    for vis_idx in range(B):
        row_idx = vis_idx // n_cols
        col_idx = vis_idx % n_cols
        im = axes[row_idx,col_idx].imshow(attn[vis_idx].cpu().numpy())

    fig.colorbar(im, ax=axes.ravel().tolist())
    plt.savefig(name+'_attn.png')
    plt.close()


def replace_nested_dict_item(obj, key, replace_value):
    for k, v in obj.items():
        if isinstance(v, dict):
            obj[k] = replace_nested_dict_item(v, key, replace_value)
    if key in obj:
        obj[key] = replace_value
    return obj


def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict

def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array
    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist
    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist
    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def memory_summary():
    vmem = psutil.virtual_memory()
    msg = (
        f">>> Currently using {vmem.percent}% of system memory "
        f"{humanize.naturalsize(vmem.used)}/{humanize.naturalsize(vmem.available)}"
    )
    print(msg)

@functools.lru_cache(maxsize=64, typed=False)
def memcache(path):
    suffix = Path(path).suffix
    print(f"loading features >>>", end=" ")
    tic = time.time()
    if suffix == ".npy":
        res = np_loader(path)
    else:
        raise ValueError(f"unknown suffix: {suffix} for path {path}")
    print(f"[Total: {time.time() - tic:.1f}s] ({socket.gethostname() + ':' + str(path)})")
    return res

def np_loader(np_path, l2norm=False):
    with open(np_path, "rb") as f:
        data = np.load(f, encoding="latin1", allow_pickle=True)
    if isinstance(data, np.ndarray) and data.size == 1:
        data = data[()]  # handle numpy dict storage convnetion
    if l2norm:
        print("L2 normalizing features")
        if isinstance(data, dict):
            for key in data:
                feats_ = data[key]
                feats_ = feats_ / max(np.linalg.norm(feats_), 1E-6)
                data[key] = feats_
        elif data.ndim == 2:
            data_norm = np.linalg.norm(data, axis=1)
            data = data / np.maximum(data_norm.reshape(-1, 1), 1E-6)
        else:
            raise ValueError("unexpected data format {}".format(type(data)))
    return data

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()
