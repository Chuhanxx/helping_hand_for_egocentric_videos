import os
import sys
import tqdm
import time
import argparse
import numpy as np
from sacred import Experiment

sys.path.insert(0,"./../")

import torch
import torch.nn.functional as F
from utils.utils import img_denorm
from model.metric import sim_matrix
from utils.parse_config import ConfigParser
from model.tfm_decoder import ObjDecoder, Cross_Attention

from collections import OrderedDict
from einops import rearrange 
import torchvision
from model.tokenizer import SimpleTokenizer
from model.LaviLa import CLIP_OPENAI_TIMESFORMER_LARGE
from data_loader.Egtea import *
import torchvision.transforms as transforms
from data_loader.lavila_transforms import Permute, SpatialCrop, TemporalCrop, Normalize
from sklearn.metrics import confusion_matrix

ex = Experiment('test')

def get_mean_accuracy(cm):
    list_acc = []
    for i in range(len(cm)):
        acc = 0
        if cm[i, :].sum() > 0:
            acc = cm[i, i] / cm[i, :].sum()
        list_acc.append(acc)

    return 100 * np.mean(list_acc), 100 * np.trace(cm) / np.sum(cm)


def get_model_card(tag):
    model_card_dict = {}
    return model_card_dict.get(tag, tag)


def inflate_positional_embeds(
    current_model_state_dict, new_state_dict,
    num_frames=4,
    load_temporal_fix='bilinear',
    name = 'visual.temporal_embed',
    dim = 1,
):
    # allow loading of timesformer with fewer num_frames
    curr_keys = list(current_model_state_dict.keys())
    if name in new_state_dict and name in curr_keys:
        load_temporal_embed = new_state_dict[name]
        load_num_frames = load_temporal_embed.shape[dim]
        curr_num_frames = num_frames
        embed_dim = load_temporal_embed.shape[-1]

        if load_num_frames != curr_num_frames:
            if load_num_frames > curr_num_frames:
                print(f'### loaded SpaceTimeTransformer model has MORE frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
            else:
                print(f'### loaded SpaceTimeTransformer model has FEWER frames than current...'
                      f'### loading weights, filling in the extras via {load_temporal_fix}')
                if load_temporal_fix == 'zeros':
                    new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                    new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                elif load_temporal_fix in ['interp', 'bilinear']:
                    # interpolate
                    # unsqueeze so pytorch thinks its an image
                    mode = 'nearest'
                    if load_temporal_fix == 'bilinear':
                        mode = 'bilinear'
                    load_temporal_embed = load_temporal_embed.unsqueeze(0)
                    new_temporal_embed = F.interpolate(load_temporal_embed,
                                                       (curr_num_frames, embed_dim), mode=mode).squeeze(0)
                else:
                    raise NotImplementedError
            new_state_dict[name] = new_temporal_embed
    # allow loading with smaller spatial patches. assumes custom border crop, to append the
    # border patches to the input sequence
    if name in new_state_dict and name in curr_keys:
        
        load_pos_embed = new_state_dict[name]
        load_num_patches = load_pos_embed.shape[dim]
        curr_pos_embed = current_model_state_dict[name]

        if load_num_patches != curr_pos_embed.shape[dim]:
            raise NotImplementedError(
                'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

    return new_state_dict

@ex.main
def run():
    tic = time.time()

    save_name = f"results/EGTEA_{args.num_frames}fx{args.num_clips}_{args.save_name}_results.pth"
    backbone = CLIP_OPENAI_TIMESFORMER_LARGE(
                pretrained=None,
                pretrained2d=False,
                text_use_cls_token=False,
                project_embed_dim=256,
                gated_xattn=False,
                random_init_gpt2=False,
                timesformer_gated_xattn=False,
                timesformer_freeze_space=False,
                freeze_lm_vclm=False,
                freeze_visual_vclm=False,
                freeze_visual_vclm_temporal=False,
                num_frames=args.num_frames,
                drop_path_rate=0,
                temperature_init=0.07,
            )

    checkpoint = torch.load(args.lavila_weights_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v

    new_state_dict = inflate_positional_embeds(
            backbone.state_dict(), new_state_dict,
            num_frames=args.num_frames,
            load_temporal_fix='bilinear',
        )
    result = backbone.load_state_dict(new_state_dict, strict=True)
    print("================Backbone Loading Result==================")
    print(result)
    feature_dim = 1024
    tokenizer = SimpleTokenizer()

    if args.method == 'xattn':
        num_queries = args.num_queries+1
        tfm = Cross_Attention(normalize_before=True,
                    return_intermediate_dec=True,)
        model = ObjDecoder(transformer=tfm, 
                    num_classes=22047, 
                    num_queries=num_queries, 
                    aux_loss=True,
                    num_frames=args.num_frames,
                    pred_traj=args.pred_traj,
                    feature_dim=feature_dim,
                    self_attn=False)
    else: 
        model = torch.nn.Identity()

    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')

        new_state_dict = checkpoint['state_dict']
        new_state_dict = inflate_positional_embeds(
            model.state_dict(), new_state_dict,
            num_frames=args.num_frames,
            load_temporal_fix='bilinear',
            name = 'temporal_embed'
        )
        try:
            model.load_state_dict(new_state_dict)
        except:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')

    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize(224),
        TemporalCrop(frames_per_clip=args.num_frames, stride=args.num_frames),
        SpatialCrop(crop_size=224, num_crops=args.num_crops),
        Normalize(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
        ])
    action_idx_file = os.path.join(args.anno_dir, 'action_idx.txt')
    label_list, label_mapping = generate_label_map(action_idx_file)
    val_files = []
    for split_i in range(1,4):
        val_files.append(os.path.join(args.anno_dir, f'test_split{split_i}.txt'))
    mean_cls_accs, accs =[],[]
    for val_file in val_files:
        val_dataset = VideoClassyDataset(
                'egtea', 
                args.video_dir, 
                val_file, 
                val_transform,
                is_training=False, 
                label_mapping=label_mapping,
                num_clips=args.num_clips,
                clip_length=args.num_frames, 
                clip_stride=2,
                sparse_sample=False,
                is_trimmed=True,
                anno_dir=args.anno_dir,
            )

        data_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, drop_last=False)

        mean_class_acc, acc = evaluate_egtea(data_loader, model, backbone, tokenizer, args, label_list, ori_sim_matrix=None, save=True, save_name=save_name)
        mean_cls_accs.append(mean_class_acc)
        accs.append(acc)
    print(f'avg_mean_class_acc:{np.mean(mean_cls_accs):.2f}. avg_acc:{np.mean(accs):.2f}')

def evaluate_egtea(data_loader, model, backbone, tokenizer, args, label_list, ori_sim_matrix=None, save=False, save_name=None):
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    backbone.eval()
    model = model.to(device)
    model.eval()

    vid_embed_arr = []
    # leave this for now since not doing anything on the gpu
    texts = tokenizer(label_list)
    text_embeds,text_feature_map= backbone.encode_text(texts.to(device))

    if args.method == 'xattn':
        sen_lens =  texts.argmax(dim=-1)
        text_embeds =  model.txt_proj(text_feature_map[np.arange(106),sen_lens])
    text_embeds = text_embeds.detach().cpu()

    labels,logits = [],[]
    with torch.no_grad():
        # for i, data in enumerate(data_loader):
        for i, data in tqdm.tqdm(enumerate(data_loader),total=len(data_loader)):
            
            frames, label = data
            labels.append(label)
            frames = torch.cat(frames).transpose(1,2).to(device) # B T C H W
            
            vid_embed, video_feature_map = backbone.encode_image(frames)
            if args.method == 'xattn':
                grid_side = int(((video_feature_map.shape[1])/args.num_frames)**0.5)
                video_grid = rearrange(video_feature_map[:, 1:, :], 'b (t h w) c -> b t (h w) c', t=args.num_frames, h=grid_side, w=grid_side)
                detr_out, hs,_,_ = model(video_grid)
                vid_embed = model.obj_proj(hs[-1, :])[:,-1,...]

            logit = sim_matrix(vid_embed.detach().cpu(), text_embeds)
            logit = logit.view(-1, args.num_clips*args.num_crops, 106).max(1)[0]
            logits.append(logit)
            if args.visualize:
                torchvision.utils.save_image(img_denorm(frames[0][0].transpose(0,1),
                                                        mean=[108.3272985, 116.7460125, 104.09373615000001], 
                                                        std=[68.5005327, 66.6321579, 70.32316305]),
                                            'egtea_vis.png')

    labels = torch.cat(labels)
    logits =  torch.cat(logits)
    cm = confusion_matrix(labels, logits.argmax(axis=1))

    mean_class_acc, acc = get_mean_accuracy(cm)
    if save:
        out_dict = {'pred':logits, 'labels': labels}
        torch.save(out_dict,save_name)
    print(f'mean_class_acc:{mean_class_acc:.2f}, acc:{acc:.2f}')    
    

    return  mean_class_acc, acc



if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')

    args.add_argument('-r', '--resume',
                      default='',
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-gpu', '--gpu', default=0, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-s', '--sliding_window_stride', default=-1, type=int,
                      help='test time temporal augmentation, repeat samples with different start times.')
    args.add_argument('-j', '--num_workers', default=8, type=int)
    args.add_argument('--num_frames', default=4, type=int)
    args.add_argument('--num_clips', default=10, type=int)
    args.add_argument('--num_crops', default=1, type=int)

    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('--backbone', default='LaviLa', type=str,
                      help='the backbone to use')
    args.add_argument('--num_queries', default=12, type=int,
                      help='number of queries in the cross transformer')
    args.add_argument('--method', default='xattn', type=str)
    args.add_argument('--pred_traj', action='store_true')
    args.add_argument('--lavila_weights_path', default='/users/czhang/work/model/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth', type=str,
                      help='path to the pretrained weights of the lavila backbone')
    args.add_argument('--anno_dir', default='', type=str,
                      help='path to directory of txt files containing annotations of EGTEA dataset')
    args.add_argument('--video_dir', default='', type=str,
                      help='path to directory of EGTEA video clips')
    args.add_argument('--save_name', default='results', type=str,
                      help='name of the file for saving results')
    args.add_argument('--visualize', action='store_true')

    config = ConfigParser(args, test=True, eval_mode='epic')

    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)

    ex.run()


# python test_egtea.py  --save_name rebuttal_fix_check  --num_queries 12 --num_frames 16   --num_clips 10 --video_dir '/scratch/shared/beegfs/htd/DATA/GTEA/cropped_clips'   --anno_dir '../data/EGTEA' --resume /users/czhang/work/EgoVLP/log-post-ddl/detr+egonce_bs128_lr3e-05_nq12_morepos_avglog_wordcontrast_vnmask_simmask06_strictmask_0_newboxes_resumebest_check/model/candidate1.pth.tar
