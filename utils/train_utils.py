import torch
from datetime import datetime
import os 
import glob
import json
import sys
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from datetime import datetime
from collections import deque

def clip_gradients(model, clip_grad=3):
    """from https://github.com/facebookresearch/dino/blob/main/main_dino.py"""
    norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            norms.append(param_norm.item())
            clip_coef = clip_grad / (param_norm + 1e-6)
            if clip_coef < 1:
                p.grad.data.mul_(clip_coef)
    return norms


def optim_policy(backbone, model, lr, wd):
    params = []
    no_decay = ['.ln_', '.bn', '.bias', '.logit_scale', '.entropy_scale']
    param_group_no_decay = []
    param_group_with_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            print(f'Param not requires_grad: {name}')
            continue
        if any([i in name for i in no_decay]):
            param_group_no_decay.append(param)
        else:
            param_group_with_decay.append(param)
    for name, param in backbone.named_parameters():
        param.requires_grad = False

    params.append({'params': param_group_no_decay, 'lr': lr, 'weight_decay': 0.0})
    params.append({'params': param_group_with_decay, 'lr': lr, 'weight_decay': wd})

    return params


def _valid_all_gather(data, n_gpu):
    """wrapper fn for all_gather, handle 1-gpu training as well."""
    if n_gpu == 1:
        return data[None,:]
    else:
        data_all = [torch.zeros_like(data) for _ in range(n_gpu)]
        torch.distributed.all_gather(data_all, data)
        data_all = torch.cat(data_all, dim=0)
        return data_all
    
def verbose(epoch, iter, metrics, name="TEST"):
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s} epoch {epoch}, iteration {iter}, {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg

def format_nested_metrics_for_writer(metrics, mode, name="TEST"):
    res = {}
    for key, val in metrics.items():
        log_name = f"[{mode}]{name}_{key}"
        res[log_name] = val
    return res

def set_path(args):
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M")
    args.launch_timestamp = dt_string
    name_prefix = f"{args.name_prefix}" if args.name_prefix else ""
    exp_path = (f"{name_prefix}"
        f"{'+'.join(args.loss)}_"
        f"bs{args.batch_size}_lr{args.lr}")
    log_path = os.path.join(exp_path, 'log')
    model_path = os.path.join(exp_path, 'model')
    if not os.path.exists(log_path) and args.rank == 0: 
        os.makedirs(log_path)
    if not os.path.exists(model_path) and args.rank == 0: 
        os.makedirs(model_path)
    if args.rank ==0:
        with open(f'{log_path}/running_command.txt', 'a') as f:
            json.dump({'command_time_stamp':dt_string, **args.__dict__}, f, indent=2)
            f.write('\n')
    return log_path, model_path, exp_path


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
