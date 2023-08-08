"""train DETR-like loss on detic results"""
import os
import numpy as np
from regex import D, P
import torch
import torch.nn as nn
import random
from tqdm import tqdm
from einops import rearrange 
import argparse
import sys
import logging
from collections import OrderedDict

sys.path.insert(0, "..")
from model.LaviLa import CLIP_OPENAI_TIMESFORMER_LARGE
from model.metric import egomcq_accuracy_metrics, sim_matrix
from model.tfm_decoder import ObjDecoder, Cross_Attention

from data_loader.data_loader import MultiDistTextVideoDataLoader
from model.tokenizer import SimpleTokenizer



@torch.no_grad()
def evaluate_egomcq(loader, model, backbone, tokenizer, device, args,):
    """
    Validate after training an epoch

    :return: A log that contains information about validation

    Note:
        The validation metrics in log must have the key 'val_metrics'.
    """
    model.eval()
    total_val_loss = [0] * len(loader)

    gt_arr = {x: [] for x in range(len(loader))}
    i2t_arr = {x: [] for x in range(len(loader))}
    t2i_arr = {x: [] for x in range(len(loader))}
    type_arr = {x: [] for x in range(len(loader))}

    save_dict = {'gt': [],
                'text_to_image': [],
                'image_to_text': [],
                'cls': [],
                'data_type': []}


    with torch.no_grad():
        # for validation we switch the nested loop order, because alternate batches not needed...
        # ... and dataloaders can be of different length
        dl_idx = 0
        for batch_idx, data in enumerate(tqdm(loader)):

            data['video'] = data['video'][0]  # remove batch
            data['text_str'] = data['text']
            data['video'] = data['video'].to(device)
            data['text'] = tokenizer(data['text']).unsqueeze(0)
            out = backbone(data['video'], data['text'].to(device),return_feature_map=True)
            video_embeds = out['image_embed'] # after projection 
            text_embeds = out['text_embed']

            video_feature_map = out['image_feature_map'] # before projection 
            text_feature_map = out['text_feature_map'] # before projection 

            # first get the video and text features from the backbone
            if args.method == 'xattn':
                grid_side = int(((video_feature_map.shape[1])/4)**0.5)
                video_grid = rearrange(video_feature_map[:, 1:, :], 'b (t h w) c -> b t (h w) c', t=4, h=grid_side, w=grid_side)
                sen_lens =  data['text'].argmax(dim=-1)
                text_embeds =  model.txt_proj(text_feature_map[0,sen_lens])
                detr_out, hs,_, _ = model(video_grid)
            if args.method == 'xattn':
                video_embeds = model.obj_proj(hs[-1, :])[:,-1,...]
            else:
                text_embeds = text_embeds[:,0,:]
                video_embeds = video_embeds[:,0,:]
            text_to_image = sim_matrix(text_embeds, video_embeds)
            image_to_text = text_to_image.t() 
    
            data_gt = data['correct'][0].to(device).unsqueeze(0)
            data_type = data['type'][0].to(device).unsqueeze(0)


            # collect stats
            save_dict['gt'].append(data_gt.cpu())
            save_dict['text_to_image'].append(text_to_image.cpu())
            save_dict['image_to_text'].append(image_to_text.cpu())
            save_dict['data_type'].append(data_type.cpu())
    
            data_gt_all = _valid_all_gather(data_gt, args.world_size)
            image_to_text_all = _valid_all_gather(image_to_text, args.world_size)
            text_to_image_all = _valid_all_gather(text_to_image, args.world_size)
            data_type_all = _valid_all_gather(data_type, args.world_size)

            gt_arr[dl_idx].append(data_gt_all.cpu())
            i2t_arr[dl_idx].append(image_to_text_all.cpu())
            t2i_arr[dl_idx].append(text_to_image_all.cpu())
            type_arr[dl_idx].append(data_type_all.cpu())


    save_dict['gt'] = torch.stack(save_dict['gt'], 0)
    save_dict['text_to_image'] = torch.stack(save_dict['text_to_image'], 0)
    save_dict['image_to_text'] = torch.stack(save_dict['image_to_text'], 0)
    save_dict['data_type'] = torch.stack(save_dict['data_type'], 0)

    nested_metrics = {x: {} for x in range(len(loader))}

    gt_arr_cat = torch.cat(gt_arr[dl_idx])
    i2t_arr_cat = torch.cat(i2t_arr[dl_idx])
    t2i_arr_cat = torch.cat(t2i_arr[dl_idx])
    type_cat = torch.cat(type_arr[dl_idx])

    metric_name = 'egomcq_accuracy_metrics'
    res_i2t = egomcq_accuracy_metrics(i2t_arr_cat, gt_arr_cat, type_cat)
    res_t2i = egomcq_accuracy_metrics(t2i_arr_cat, gt_arr_cat, type_cat)

    torch.save(save_dict, f'{args.log_path}/EgoMCQ_results{args.results_suffix}.pth')
    nested_metrics[dl_idx][metric_name+'_i2t'] = res_i2t
    nested_metrics[dl_idx][metric_name+'_t2i'] = res_t2i

    logging.info(
        verbose(metrics=res_i2t, name='i2t'))
    logging.info(
        verbose(metrics=res_t2i, name='ti2'))
        
    res_dict = {}

    if args.rank == 0:
        res_dict = {}
        res_dict['nested_val_metrics'] = nested_metrics
        res_dict['t2i_acc'] = res_t2i

    return res_dict


def verbose(metrics, name="TEST"):
    msg = ""
    for key in metrics.keys():
        acc = metrics[key]
        msg += f"{name:s}  {key:s}, Acc: {acc:.1f};    "
    print(msg)
    return msg

def _valid_all_gather(data, n_gpu):
    """wrapper fn for all_gather, handle 1-gpu training as well."""
    if n_gpu == 1:
        return data[None,:]
    else:
        data_all = [torch.zeros_like(data) for _ in range(n_gpu)]
        torch.distributed.all_gather(data_all, data)
        data_all = torch.cat(data_all, dim=0)
        return data_all
    


def setup(args):
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    if args.world_size >1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        device = torch.device(f'cuda:{args.local_rank}')
        if args.rank == 0:
            print('world_size', args.world_size, flush=True)
            print('local_rank: ', args.local_rank, flush=True)
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        num_gpu = args.world_size
        args.num_gpu = num_gpu
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.iteration = 1

    if '/srun' in os.environ['_']:  # in sbatch
        print('running command: {')
        for key, item in args.__dict__.items():
            print(f'  "{key}": {item}')
        print('}')

    return device


def get_model_card(tag):
    model_card_dict = {}
    return model_card_dict.get(tag, tag)



def main(args):
    device = setup(args)

    ### model ###
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
                num_frames=4,
                drop_path_rate=0,
                temperature_init=0.07,
                use_adapter=False,
            )
    feature_dim = 1024
    checkpoint = torch.load(args.lavila_weights_path, map_location='cpu')
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
        
    result = backbone.load_state_dict(new_state_dict, strict=True)
    print("================Backbone Loading Result==================")
    print(result)
    tsfm_params = {
        "force_centercrop": True,
        "norm_mean": [108.3272985/255, 116.7460125/255, 104.09373615000001/255],
        "norm_std":[68.5005327/255, 66.6321579/255, 70.32316305/255]}
    
    if args.method == 'xattn':  
        num_queries = args.num_queries+1
        tfm = Cross_Attention(normalize_before=True,
                       return_intermediate_dec=True,)
        model = ObjDecoder(transformer=tfm, 
                     num_classes=args.num_classes, 
                     num_queries=num_queries, 
                     aux_loss=True,
                     pred_traj=True,
                     feature_dim=feature_dim,
                     self_attn=False)
    else:
        model = nn.Identity()

    model.to(device)
    model_without_dp = model

    text_params={"input": "text"}
    video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
    }
    data_dir=os.path.join(args.data_dir,f"videos_{args.raw_resolution}_chunked")
    val_loader = MultiDistTextVideoDataLoader(args,
                                        'EgoClip',
                                        text_params,
                                        video_params,
                                        data_dir,
                                        meta_dir=args.meta_dir,
                                        split='val',
                                        batch_size=1,  
                                        video_res=args.video_res,
                                        num_workers=args.num_workers,
                                        reader='cv2_egoclip',
                                        shuffle=False, # double check
                                        subsample='mcq',
                                        neg_param=False,
                                        tsfm_params= tsfm_params,
                                        crop_w_boxes=False,
                                        )
    

    backbone.to(device)
    ### restart ###
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=False)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training, continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()


    tokenizer = SimpleTokenizer()
    val_loss = evaluate_egomcq(val_loader, model, backbone, tokenizer, device, args)
    sys.exit(0)


def parse_args():
    try:    # with ddp
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    except:  # for debug only
        world_size = 1
        rank = 0
        local_rank = 0

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', default='xattn', type=str)
    parser.add_argument('--seed', default=888,type=int)

    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gpu', default=None, type=str)
    parser.add_argument('-j', '--num_workers', default=8, type=int)
    parser.add_argument('--log_path', default='results/', type=str)
    parser.add_argument('--data_dir', default='', type=str)

    parser.add_argument('--backbone', default='LaviLa', type=str)
    parser.add_argument('--results_suffix', default='', type=str)
    parser.add_argument('--num_classes', default=22047, type=int, help='a fake number, not used')
    parser.add_argument('--lavila_weights_path', default='/users/czhang/work/model/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth', type=str,
                      help='path to the pretrained weights of the lavila backbone')
    parser.add_argument('--meta_dir', default='', type=str)
    parser.add_argument('--num_frames', default=4, type=int)

    parser.add_argument('--video_res', default=224, type=int)
    parser.add_argument('--num_queries', default=12, type=int)
    parser.add_argument('--raw_resolution', default=256, type=int)

    parser.add_argument('-k', '--local_rank', type=int, default=local_rank)
    parser.add_argument('-ws', '--world_size', type=int, default=world_size)
    parser.add_argument('-rk', '--rank', type=int, default=rank)
    args = parser.parse_args()
    return args





if __name__ == '__main__':
    args = parse_args()
    main(args)


# python test_EgoMCQ.py   --data_dir '/scratch/shared/beegfs/htd/DATA/ego4d/'  --meta_dir '../data/EgoClip' --resume /users/czhang/work/EgoVLP/log-post-ddl/detr+egonce_bs128_lr3e-05_nq12_morepos_avglog_wordcontrast_vnmask_simmask06_strictmask_0_newboxes_resumebest_check/model/candidate1.pth.tar  
