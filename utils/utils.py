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



def save_runtime_checkpoint(state, filename, rm_history=True):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    assert filename.endswith('.pth.tar')
    torch.save(state, filename.replace('.pth.tar', f'_{dt_string}.pth.tar'))
    if rm_history:
        history = sorted(glob.glob(filename.replace('.pth.tar', '_*.pth.tar')))
        if len(history) > 10:
            try:
                history = history[:-10]
                for h in history:os.remove(h)
            except:
                print(f'Caught Error when saving runtime checkpoint: {sys.exc_info()[0]}')
                pass
     

def save_checkpoint(state, is_best=0, filename='models/checkpoint.pth.tar', keep_all=False):
    torch.save(state, filename)
    last_epoch_path = os.path.join(os.path.dirname(filename),
                                   'epoch%s.pth.tar' % str(state['epoch']))

    if not keep_all:
        try: os.remove(last_epoch_path)
        except: pass

    if is_best:
        past_best = glob.glob(os.path.join(os.path.dirname(filename), 'model_best_*.pth.tar'))
        past_best = sorted(past_best, key=lambda x: int(''.join(filter(str.isdigit, x))))
        if len(past_best) >= 10:
            try: os.remove(past_best[0])
            except: pass
        # for i in past_best:
        #     try: os.remove(i)
        #     except: pass
        torch.save(state, os.path.join(os.path.dirname(filename), 'model_best_epoch%s.pth.tar' % str(state['epoch'])))
    # old (from pytorch example)
    # if is_best:
        # shutil.copyfile(filename, os.path.join(os.path.dirname(filename), 'model_best.pth.tar'))

def write_log(content, epoch, filename):
    if not os.path.exists(filename):
        log_file = open(filename, 'w')
    else:
        log_file = open(filename, 'a')
    log_file.write('## Epoch %d:\n' % epoch)
    log_file.write('time: %s\n' % str(datetime.now()))
    log_file.write(content + '\n\n')
    log_file.close()

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


def calc_topk_accuracy(output, target, topk=(1,), accept_short=False):
    """
    Modified from: https://gist.github.com/agermanidis/275b23ad7a10ee89adccf021536bb97e
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    try:
        _, pred = output.topk(maxk, 1, True, True)
    except Exception as e:
        if accept_short:
            maxk = output.size(1)
            _, pred = output.topk(maxk, 1, True, True)
        else:
            raise e

    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res

def calc_mask_accuracy(output, target_mask, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk,1,True,True)

    zeros = torch.zeros_like(target_mask).long()
    pred_mask = torch.zeros_like(target_mask).long()

    res = []
    for k in range(maxk):
        pred_ = pred[:,k].unsqueeze(1)
        onehot = zeros.scatter(1,pred_,1)
        pred_mask = onehot + pred_mask # accumulate 
        if k+1 in topk:
            res.append(((pred_mask * target_mask).sum(1)>=1).float().mean(0))
    return res 

def calc_multi_accuracy(output, target_mask, num_class, topk=(1,)):
    '''support multiple (num_class) correct answer in bool target_mask'''
    maxk = max(topk)
    batch_size = output.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    _, target = target_mask.topk(num_class, 1, True, True)
    pred = pred.view(batch_size, 1, maxk).expand(batch_size, num_class, maxk)
    target = target.view(batch_size, num_class, 1).expand_as(pred)
    correct = pred.eq(target).sum(1).t().contiguous()

    res = []
    for k in topk:
        # if one of topN index matches one of target idx, consider to be correct 
        correct_k, _ = correct[:k].max(0)
        correct_k = correct_k.view(-1).float().sum(0)
        res.append(correct_k.mul_(1 / batch_size))
    return res


def strfdelta(tdelta, fmt):
    d = {"d": tdelta.days}
    d["h"], rem = divmod(tdelta.seconds, 3600)
    d["m"], d["s"] = divmod(rem, 60)
    return fmt.format(**d)

def get_ap(binary_list):
    length = len(binary_list)
    total_tp = sum(binary_list)
    if total_tp == 0: return 0

    result = 0
    tp_count = 0
    for idx in range(length):
        now = binary_list[idx]
        if now == 1:
            tp_count += 1
            result += tp_count / (idx+1)
        else:
            pass 
    return result / total_tp

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



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name='null', fmt=':.4f'):
        self.name = name 
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.local_history = deque([])
        self.local_avg = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, n=1, history=0, step=5):
        self.val = val
        self.sum += val * n
        self.count += n
        if n == 0: return
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)
        if step > 0:
            self.local_history.append(val)
            if len(self.local_history) > step:
                self.local_history.popleft()
            self.local_avg = np.average(self.local_history, 0)


    def dict_update(self, val, key):
        if key in self.dict.keys():
            self.dict[key].append(val)
        else:
            self.dict[key] = [val]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = self.dict[key]
            avg_val = np.average(val)
            len_val = len(val)
            std_val = np.std(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val, std_val])
            else:
                self.save_dict[key] = [[avg_val, std_val]]

            print('Activity:%s, mean %s is %0.4f, std %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, title, std_val, len_val))

            total.extend(val)

        self.dict = {}
        avg_total = np.average(total)
        len_total = len(total)
        std_total = np.std(total)
        print('\nOverall: mean %s is %0.4f, std %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, title, std_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f)

    def __len__(self):
        return self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter_Weighted(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.base = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.history = []
        self.dict = {} # save all data values here
        self.save_dict = {} # save mean and std here, for summary table

    def update(self, val, base, n=1, history=0):
        self.val = val
        self.base = base
        self.sum += val * base * n
        self.count += base * n
        self.avg = self.sum / self.count
        if history:
            self.history.append(val)

    def dict_update(self, val, base, key):
        if key in self.dict.keys():
            self.dict[key].append([val, base])
        else:
            self.dict[key] = [[val, base]]

    def print_dict(self, title='IoU', save_data=False):
        """Print summary, clear self.dict and save mean+std in self.save_dict"""
        total = []
        for key in self.dict.keys():
            val = np.array(self.dict[key])
            avg_val = np.sum(val[:,0] * val[:,1]) / np.sum(val[:,1])
            len_val = len(val)

            if key in self.save_dict.keys():
                self.save_dict[key].append([avg_val])
            else:
                self.save_dict[key] = [[avg_val]]

            print('Activity:%s, mean %s is %0.4f, length of data is %d' \
                % (key, title, avg_val, len_val))

            total.extend(val)

        self.dict = {}
        total = np.array(total)
        avg_total = np.sum(total[:,0] * total[:,1]) / np.sum(total[:,1])
        len_total = len(total)
        print('\nOverall: mean %s is %0.4f, length of data is %d \n' \
            % (title, avg_total, len_total))

        if save_data:
            print('Save %s pickle file' % title)
            with open('img/%s.pickle' % title, 'wb') as f:
                pickle.dump(self.save_dict, f)

    def __len__(self):
        return self.count

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

class Plotter(object):
    """plot loss and accuracy, require import matplotlib.pyplot as plt"""
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def train_update(self, loss, acc):
        if type(loss) != float:
            loss = float(loss)
        if type(acc) != float:
            acc = float(acc)
        self.train_loss.append(loss)
        self.train_acc.append(acc)

    def val_update(self, loss, acc):
        if type(loss) != float:
            loss = float(loss)
        if type(acc) != float:
            acc = float(acc)
        self.val_loss.append(loss)
        self.val_acc.append(acc)

    def export_valacc(self, filename):
        pickle.dump(self.val_acc, open(filename+'.pickle', 'wb'))

    def draw(self, filename, title=['loss', 'acc']):
        if len(self.val_loss) == len(self.train_loss):
            zipdata = zip([[self.train_loss, self.val_loss],
                           [self.train_acc, self.val_acc]],
                          title)
            for i, (data, name) in enumerate(zipdata):
                plt.subplot(1, 2, i+1)
                plt.plot(data[0], label='train')
                plt.plot(data[1], label='val')
                plt.legend(loc='upper left')
                plt.xlabel('epoch')
                plt.grid()
                plt.title(name)
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()
        else: # no validation data
            zipdata = zip([self.train_loss, self.train_acc],
                          title)
            for i, (data, name) in enumerate(zipdata):
                plt.subplot(1, 2, i+1)
                plt.plot(data, label='train')
                plt.legend(loc='upper left')
                plt.xlabel('iteration')
                if name == 'loss':
                    try:
                        plt.ylim(0, max(np.mean(data)*2, 1))
                    except:
                        pass
                plt.grid()
                plt.title(name)
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()


class Plotter_split(object):
    """plot accuracy, hard and negative, 
    require import matplotlib.pyplot as plt"""
    def __init__(self):
        self.train_acc_hard = []
        self.train_acc_easy = []
        self.val_acc_hard = []
        self.val_acc_easy = []

    def train_update(self, acc_hard, acc_easy):
        if type(acc_hard) != float:
            acc_hard = float(acc_hard)
        if type(acc_easy) != float:
            acc_easy = float(acc_easy)
        self.train_acc_hard.append(acc_hard)
        self.train_acc_easy.append(acc_easy)

    def val_update(self, acc_hard, acc_easy):
        if type(acc_hard) != float:
            acc_hard = float(acc_hard)
        if type(acc_easy) != float:
            acc_easy = float(acc_easy)
        self.val_acc_hard.append(acc_hard)
        self.val_acc_easy.append(acc_easy)

    def draw(self, filename, title=['acc_hard', 'acc_easy']):
        if len(self.val_acc_hard) == len(self.train_acc_hard):
            zipdata = zip([[self.train_acc_hard, self.val_acc_hard],
                           [self.train_acc_easy, self.val_acc_easy]],
                            title)
            for i, (data, name) in enumerate(zipdata):
                plt.subplot(1, 2, i+1)
                plt.plot(data[0], label='train')
                plt.plot(data[1], label='val')
                plt.legend(loc='upper left')
                plt.xlabel('epoch')
                plt.grid()
                plt.title(name)
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()
        else: # no validation data
            zipdata = zip([self.train_acc_hard, self.train_acc_easy],
                          title)
            for i, (data, name) in enumerate(zipdata):
                plt.subplot(1, 2, i+1)
                plt.plot(data, label='train')
                plt.legend(loc='upper left')
                plt.xlabel('iteration')
                plt.grid()
                plt.title(name)
            plt.tight_layout()
            plt.savefig(filename)
            plt.clf()


class AccuracyTable(object):
    """compute accuracy for each class"""
    def __init__(self):
        self.dict = {}

    def update(self, pred, tar):
        pred = torch.squeeze(pred)
        tar = torch.squeeze(tar)
        for i, j in zip(pred, tar):
            i = int(i)
            j = int(j)
            if j not in self.dict.keys():
                self.dict[j] = {'count':0,'correct':0}
            self.dict[j]['count'] += 1
            if i == j:
                self.dict[j]['correct'] += 1

    def print_table(self, label):
        for key in self.dict.keys():
            acc = self.dict[key]['correct'] / self.dict[key]['count']
            print('%s: %2d, accuracy: %3d/%3d = %0.6f' \
                % (label, key, self.dict[key]['correct'], self.dict[key]['count'], acc))


class AverageMeter_Raw(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ConfusionMeter(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.mat = np.zeros((num_class, num_class)).astype(np.int)
        self.precision = []
        self.recall = []

    def update(self, pred, tar):
        pred, tar = pred.cpu().numpy(), tar.cpu().numpy()
        pred = np.squeeze(pred)
        tar = np.squeeze(tar)
        for p,t in tqdm(zip(pred.flat, tar.flat), total=len(pred.flat), disable=True):
            self.mat[p][t] += 1

    def print_mat(self):
        print('Confusion Matrix: (target in columns)')
        print(self.mat)

    def plot_mat(self, path, dictionary=None, annotate=False):
        plt.figure(dpi=600)
        plt.imshow(self.mat,
            cmap=plt.cm.jet,
            interpolation=None,
            extent=(0.5, np.shape(self.mat)[0]+0.5, np.shape(self.mat)[1]+0.5, 0.5))
        width, height = self.mat.shape
        if annotate:
            for x in range(width):
                for y in range(height):
                    plt.annotate(str(int(self.mat[x][y])), xy=(y+1, x+1),
                                 horizontalalignment='center',
                                 verticalalignment='center',
                                 fontsize=8)

        if dictionary is not None:
            plt.xticks([i+1 for i in range(width)],
                       [dictionary[i] for i in range(width)],
                       rotation='vertical')
            plt.yticks([i+1 for i in range(height)],
                       [dictionary[i] for i in range(height)])
        plt.xlabel('Ground Truth')
        plt.ylabel('Prediction')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(path, format='svg')
        plt.clf()

        # for i in range(width):
        #     if np.sum(self.mat[i,:]) != 0:
        #         self.precision.append(self.mat[i,i] / np.sum(self.mat[i,:]))
        #     if np.sum(self.mat[:,i]) != 0:
        #         self.recall.append(self.mat[i,i] / np.sum(self.mat[:,i]))
        # print('Average Precision: %0.4f' % np.mean(self.precision))
        # print('Average Recall: %0.4f' % np.mean(self.recall))


class LabelQueue(object):
    def __init__(self, length=10, mode='max'):
        self.queue = []
        self.label = []
        self.length = length
        self.max_label = 0
        self.min_label = 0
        self.mode = mode
        assert self.mode in ['max', 'min']

    def update(self, label, val):
        if len(self.label) < self.length: # append to the queue
            self._append(label, val)
        else: # update the queue
            if self.mode == 'max':
                if label > self.min_label:
                    self.label.pop(self.min_id); self.queue.pop(self.min_id)
                    self._append(label, val)
            elif self.mode == 'min':
                if label < self.max_label:
                    self.label.pop(self.max_id); self.queue.pop(self.max_id)
                    self._append(label, val)

    def _append(self, label, val):
        assert len(self.queue) == len(self.label)
        self.queue.append(val)
        self.label.append(label)
        self.max_label = np.max(self.label)
        self.max_id = np.argmax(self.label)
        self.min_label = np.min(self.label)
        self.min_id = np.argmin(self.label)

    def get_value(self):
        return self.label, self.queue

    def __len__(self):
        return len(self.label)


class Counter2d(object):
    def __init__(self, num_class, max_idx, queue_len=100):
        self.num_class = num_class
        self.max_idx = max_idx 
        self.queue_len = queue_len 
        self.reset()

    def update(self, value, index):
        assert self.counter[index].shape == value.shape
        self.counter[index] += value 

        tmp = np.zeros((self.num_class, self.max_idx))
        tmp[index] += value
        self.queue.append(tmp)
        if len(self.queue) > self.queue_len: self.queue.pop(0)

    def get_freq(self, counter=None):
        if counter is None: counter = self.counter 
        eps = 1e-8 # avoid divide by 0 
        s = counter.sum(1, keepdims=True)
        return counter / s

    def get_recent_counter(self):
        counter = np.stack(self.queue, 0).sum(0)
        assert counter.shape == (self.num_class, self.max_idx)
        return counter 

    def reset(self):
        self.counter = np.zeros((self.num_class, self.max_idx))
        self.queue = []


class GroupedSampler(torch.utils.data.Sampler):
    '''group sample by label, must work with a batch sampler'''
    def __init__(self, data_source):
        self.video_info = data_source.video_info
        self.encode_action = data_source.encode_action
        print('initializing data sampler')
        self.generate_idx_pool()
        self.generate_idx_dict(n_repeat=4)
        self.generate_idx_list(batch_size=4, repeat=1)

    def __iter__(self):
        n = len(self.idx_list)
        return iter(self.idx_list)

    def __len__(self):
        return len(self.idx_list)

    def generate_idx_pool(self):
        idx_pool = {}
        for idx, (_, row) in tqdm(enumerate(self.video_info.iterrows()), total=len(self.video_info)):
            *_, vname = row
            try:
                # vname = vpath.split('/')[-3]
                vid = self.encode_action(vname)
            except:
                # vname = vpath.split('/')[-2]
                vid = self.encode_action(vname)
            if vid not in idx_pool: idx_pool[vid] = []
            idx_pool[vid].append(idx)
        self.idx_pool = idx_pool

    def generate_idx_dict(self, n_repeat, verbose=False):
        idx_dict = {}
        omit_class = []
        for k, v in self.idx_pool.items():
            np.random.shuffle(v)
            max_idx = math.floor(len(v) / n_repeat) * n_repeat
            if max_idx == 0: omit_class.append(k); continue
            v = v[0:max_idx] # remove last a few items (if needed)
            v = np.array_split(v, math.ceil(len(v)/n_repeat))
            idx_dict[k] = v
        if verbose: print('following class id not sampled:', omit_class)
        self.idx_dict = idx_dict

    def generate_idx_list(self, batch_size, repeat=1, verbose=False):
        global_idx_list = []
        for rep in range(repeat):
            idx_list = []
            tmp_idx_dict = copy.deepcopy(self.idx_dict)
            while True: 
                if (len(tmp_idx_dict.keys()) < batch_size): break
                select_idx = np.random.choice(list(tmp_idx_dict.keys()), size=batch_size, replace=False)
                batch = []
                for i in select_idx:
                    batch.extend(tmp_idx_dict[i].pop(0))
                    if tmp_idx_dict[i].__len__() == 0: del tmp_idx_dict[i]
                # np.random.shuffle(batch)
                idx_list.append(batch)
            np.random.shuffle(idx_list)
            global_idx_list.extend(idx_list)
        self.idx_list = np.array(global_idx_list).flatten()


class GroupedBatchSampler(torch.utils.data.Sampler):
    '''generate batch, must used with GroupedSampler'''
    def __init__(self, sampler, batch_size, drop_last, group_size=8, num_gpu=1, repeat=1):
        if not isinstance(sampler, torch.utils.data.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integeral value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.group_size = group_size
        self.repeat = repeat
        self.num_gpu = num_gpu

    def __iter__(self):
        self.sampler.generate_idx_dict(n_repeat=self.group_size, verbose=True)
        self.sampler.generate_idx_list(batch_size=self.batch_size//(self.group_size*self.num_gpu), repeat=self.repeat)
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size


class CustomizedRandomSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.len = len(self.data_source)
        self.generate_idx_list(repeat=1)

    def __iter__(self):
        n = len(self.idx_list)
        return iter(self.idx_list)

    def generate_idx_list(self, repeat=1):
        global_idx_list = []
        for _ in range(repeat):
            global_idx_list.extend(torch.randperm(self.len).tolist())
        self.idx_list = global_idx_list

    def __len__(self):
        return len(self.idx_list)


class CustomizedBatchSampler(torch.utils.data.Sampler):
    def __init__(self, sampler, batch_size, drop_last, repeat=1):
        if not isinstance(sampler, torch.utils.data.Sampler):
            raise ValueError("sampler should be an instance of "
                             "torch.utils.data.Sampler, but got sampler={}"
                             .format(sampler))
        if not isinstance(batch_size, _int_classes) or isinstance(batch_size, bool) or \
                batch_size <= 0:
            raise ValueError("batch_size should be a positive integer value, "
                             "but got batch_size={}".format(batch_size))
        if not isinstance(drop_last, bool):
            raise ValueError("drop_last should be a boolean value, but got "
                             "drop_last={}".format(drop_last))
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.repeat = repeat 

    def __iter__(self):
        self.sampler.generate_idx_list(repeat=self.repeat)
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size



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
):

    # totensor = torchvision.transforms.PILToTensor()
    topil = torchvision.transforms.ToPILImage()

    # mean = torch.tensor(mean)[:,None,None].to(tensor.device)
    # std = torch.tensor(std)[:,None,None].to(tensor.device)
    # tensor = tensor.mul(std).add(mean)
    # tensor = tensor.detach()
    if norm:
        tensor = img_denorm(tensor, mean=mean, std=std)
    im = topil(tensor)
    

    colors = ['blue','green','orange','brown','pink','white','Cyan','gold','Khaki','Indigo','LightBlue','LightSalmon','SlateGray','Chocolate','DarkBlue','DarkGray','DarkSlateGrey','linen','gray','beige']

    for i,bbox in enumerate(bboxes):
        if bbox.sum() == 0 :
            continue
        draw = ImageDraw.Draw(im)
        draw.rectangle(bbox.tolist(), outline=colors[i], width=width)
        if texts is not None:
            try:
                font = ImageFont.truetype('/users/czhang/EgoVLP/miscs/arial.ttf', size=16)
                draw.text((bbox[0], max(0, bbox[1]-20)), texts[i], font=font, fill=colors[i]) 
                im.save('text.png')
            except:
                import ipdb; ipdb.set_trace()
                
    return im 

def draw_box_on_clip(bbox, frames, word=None, name = 'bbox'):
    vis_imgs = []
    for f_i in range(frames.shape[0]):
        vis_img = draw_bbox(frames[f_i], 
            bbox[f_i]*224,
            norm =False, width=3, texts=word)
        vis_imgs.append(np.array(vis_img))
    vis_img = np.concatenate(vis_imgs,1)
    vis_img = Image.fromarray(vis_img)
    vis_img.save(f'{name}.png')
    


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
