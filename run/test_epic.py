import os
import sys
import tqdm
import pickle
import argparse
import numpy as np
import pandas as pd
from sacred import Experiment

sys.path.insert(0,"./../")

import torch
import torch.nn.functional as F
from utils import nDCG, mAP
import data_loader.data_loader as module_data
from utils.utils import draw_bbox, img_denorm

from utils.parse_config import ConfigParser
from model.tfm_decoder import ObjDecoder, Cross_Attention
from einops import rearrange 
import torchvision
from model.metric import sim_matrix
from utils.box_ops import box_cxcywh_to_xyxy
from PIL import Image
from model.tokenizer import SimpleTokenizer
from model.LaviLa import CLIP_OPENAI_TIMESFORMER_LARGE
from run.test_egtea import inflate_positional_embeds
from collections import OrderedDict
import time

ex = Experiment('test')


def get_model_card(tag):
    model_card_dict = {}
    return model_card_dict.get(tag, tag)


def softmax_numpy(sim, dim=0):
    sim = torch.Tensor(sim)
    sim = F.softmax(sim, dim=dim)
    return sim.numpy()

def initialise_nDCG_values(relevancy_matrix):
    vis_k_counts = nDCG.calculate_k_counts(relevancy_matrix)
    txt_k_counts = nDCG.calculate_k_counts(relevancy_matrix.T)

    vis_IDCG = nDCG.calculate_IDCG(relevancy_matrix, vis_k_counts)
    txt_IDCG = nDCG.calculate_IDCG(relevancy_matrix.T, txt_k_counts)

    k_counts_dict = {'v': vis_k_counts, 't': txt_k_counts}
    IDCG_dict = {'v': vis_IDCG, 't': txt_IDCG}

    return IDCG_dict, k_counts_dict

def initialise_jpose_nDCG_values(relevancy_matrix):
    action_IDCG, action_k_values = initialise_nDCG_values(relevancy_matrix)

    dataset = {}
    dataset['action'] = {}
    dataset['action']['IDCG'] = action_IDCG
    dataset['action']['k_values'] = action_k_values
    return dataset

def compute_similarity_matrix(similarity_matrix,indexes,dual_softmax=False):
    print(similarity_matrix.max(),similarity_matrix.min())
    similarity_matrix = (similarity_matrix + 1) / 2
    return similarity_matrix.T[:, indexes]
@ex.main
def run():

    if args.backbone =='LaviLa':
        tsfm_params = {
            "force_centercrop": False,
            "norm_mean": [108.3272985/255, 116.7460125/255, 104.09373615000001/255],
            "norm_std":[68.5005327/255, 66.6321579/255, 70.32316305/255],
            "resize_wo_crop": True,}
    else:
        tsfm_params = None
    # setup data_loader instances
    config._config['data_loader']['type'] = 'TextVideoDataLoader'
    config._config['data_loader']['args']['split'] = args.split
    config._config['data_loader']['args']['tsfm_split'] = 'test'  # set transform to test split to remove augmentations
    config._config['data_loader']['args']['shuffle'] = False
    config._config['data_loader']['args']['batch_size'] = args.batch_size
    config._config['data_loader']['args']['sliding_window_stride'] = args.sliding_window_stride
    config._config['data_loader']['args']['tsfm_params'] = tsfm_params
    config._config['data_loader']['args']['num_workers'] = args.num_workers
    config._config['data_loader']['args']['crop_w_boxes'] = False # whether to crop the images according to bbox information
    config._config['data_loader']['args']['video_params']['num_frames'] = args.num_frames
    config._config['data_loader']['args']['data_dir']= args.data_dir
    config._config['data_loader']['args']['meta_dir']= args.meta_dir


    tic = time.time()

    out_path = f"./results/EpicKitchens_MIR_{args.num_frames}f_"+args.save_name+'.pth'
    data_loader = config.initialize('data_loader', module_data)
    path_relevancy = os.path.join(args.meta_dir, "relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl")
    relevancy = pickle.load(open(path_relevancy, 'rb'))
    indexes = pickle.load(open( os.path.join(args.meta_dir,'indexes.pkl'),'rb'))

    print('loading metadata:', time.time()-tic)
    tic = time.time()

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
    print('loading backbone:', time.time()-tic)
    tic = time.time()


    num_queries = args.num_queries+1
 

    if args.method == 'xattn':
        tfm = Cross_Attention(normalize_before=True,
                    return_intermediate_dec=True,)
        model = ObjDecoder(transformer=tfm, 
                    num_classes=22047, 
                    num_queries=num_queries, 
                    aux_loss=True,
                    num_frames=args.num_frames,
                    pred_traj=False,
                    feature_dim=feature_dim,
                    self_attn=False)
    else: 
        model = torch.nn.Identity()
    print('loading model:', time.time()-tic)
    tic = time.time()


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

    print('Resume model:', time.time()-tic)
    tic = time.time()
    avg_mAP,avg_nDCG = evaluate_ek(data_loader, model, backbone, tokenizer, relevancy, indexes, args,  out_path=out_path)

def evaluate_ek(data_loader, model, backbone, tokenizer,relevancy, indexes, args, out_path=None, argmax=None):
    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbone = backbone.to(device)
    backbone.eval()
    model = model.to(device)
    model.eval()

    text_embed_arr = []
    vid_embed_arr = []

    with torch.no_grad():
        # for i, data in enumerate(data_loader):
        for i, data in tqdm.tqdm(enumerate(data_loader)):
            # leave this for now since not doing anything on the gpu
            data['text_str'] = data['text']
            if isinstance(data['video'], list):
                data['video'] = [x.to(device) for x in data['video']]
            else:
                data['video'] = data['video'].to(device)

            data['text'] = tokenizer(data['text']).unsqueeze(0)
            out = backbone(data['video'], data['text'].to(device),return_feature_map=True)
            vid_embed = out['image_embed'] # after projection 
            text_embed = out['text_embed'] 

            video_feature_map = out['image_feature_map'] # before projection 
            text_feature_map = out['text_feature_map'] # before projection 

            grid_side = int(((video_feature_map.shape[1])/args.num_frames)**0.5)
            video_grid = rearrange(video_feature_map[:, 1:, :], 'b (t h w) c -> b t (h w) c', t=args.num_frames, h=grid_side, w=grid_side)
 

            if args.method == 'xattn':
                sen_lens =  data['text'].argmax(dim=-1)
                text_embed =  model.txt_proj(text_feature_map[0,sen_lens])
                detr_out, hs,_, _ = model(video_grid)
                vid_embed = model.obj_proj(hs[-1, :])[:,-1,...]
            else:
                vid_embed = vid_embed[:,0,:]
                text_embed = text_embed[:,0,:]

            vid_embed_arr.append(vid_embed.cpu().detach())
            text_embed_arr.append(text_embed.cpu().detach())

            if args.visualize and argmax[i]>100:
                vis_boxes = detr_out["pred_boxes"].view(1,4,-1,4)
                vis_hand_imgs, vis_obj_imgs = [],[]

                for f_i in range(4):
                    vis_img = draw_bbox(data['video'][0][f_i], box_cxcywh_to_xyxy(vis_boxes[0][f_i][:2]*224),)
                    vis_hand_imgs.append(np.array(vis_img))
                    vis_img = draw_bbox(data['video'][0][f_i], box_cxcywh_to_xyxy(vis_boxes[0][f_i][2:]*224),)
                    vis_obj_imgs.append(np.array(vis_img))
                vis_img = np.concatenate([np.concatenate(vis_hand_imgs,1), np.concatenate(vis_obj_imgs,1)],0)
                vis_img = Image.fromarray(vis_img)
                vis_img.save(f'ek_vis/ekboxes_{i}.png')
                torchvision.utils.save_image(
                    img_denorm((data['video'][0]),mean=[108.3272985/255, 116.7460125/255, 104.09373615000001/255], std=[68.5005327/255, 66.6321579/255, 70.32316305/255] ),
                    f'ek_vis/ek_{i}.png')
                pred_top5 = np.argsort(sim[i,:])[::-1][:5]
                gt_top5 = np.argsort(relevancy[i,:])[::-1][:5]

                print(pred_top5)
                print(gt_top5)
                print('Preidcted Top5:')
                for j in pred_top5:
                    print(all_texts[j])
                print('-----'*10)
                print(data['text_str'])
    
                import ipdb; ipdb.set_trace()

    vid_embeds = torch.cat(vid_embed_arr)
    text_embeds = torch.cat(text_embed_arr)
    similarity_matrix = sim_matrix(text_embeds, vid_embeds).detach().numpy()
    if out_path is not None:
        out_dict = {'pred':similarity_matrix}
        torch.save(out_dict, out_path)
    similarity_matrix = compute_similarity_matrix(similarity_matrix,indexes)
    dataset = initialise_jpose_nDCG_values(relevancy)
    vis_nDCG = nDCG.calculate_nDCG(similarity_matrix,
                                relevancy, dataset['action']['k_values']['v'],
                                IDCG=dataset['action']['IDCG']['v'])
    txt_nDCG = nDCG.calculate_nDCG(similarity_matrix.T,
                                relevancy.T, dataset['action']['k_values']['t'],
                                IDCG=dataset['action']['IDCG']['t'])
    avg_nDCG =  (vis_nDCG + txt_nDCG) / 2
    print('nDCG: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_nDCG, txt_nDCG, avg_nDCG))

    vis_mAP = mAP.calculate_mAP(similarity_matrix, relevancy)
    txt_mAP = mAP.calculate_mAP(similarity_matrix.T, relevancy.T)
    avg_mAP = (vis_mAP + txt_mAP) / 2
    print('mAP: VT:{:.3f} TV:{:.3f} AVG:{:.3f}'.format(vis_mAP, txt_mAP, avg_mAP))


    return avg_mAP,avg_nDCG


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
    args.add_argument('--data_dir', default='', type=str)

    args.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                      help='split to evaluate on.')
    args.add_argument('--batch_size', default=1, type=int,
                      help='size of batch')
    args.add_argument('--save_name', default='results', type=str,
                      help='name of the file for saving results')
    args.add_argument('--meta_dir', default='./data', type=str,
                      help='name of the file for annotation files')
    args.add_argument('--backbone', default='LaviLa', type=str,
                      help='the backbone to use')
    args.add_argument('--num_queries', default=8, type=int,
                      help='number of queries in the cross transformer')
    args.add_argument('--lavila_weights_path', default='/users/czhang/work/model/clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth', type=str,
                      help='path to the pretrained weights of the lavila backbone')
    args.add_argument('--method', default='xattn', type=str)
    args.add_argument('--visualize', action='store_true')
    args.add_argument('--num_frames', default=4, type=int)

    config = ConfigParser(args, test=True, eval_mode='epic')

    # hack to get sliding into config
    args = args.parse_args()
    config._config['sliding_window_stride'] = args.sliding_window_stride
    ex.add_config(config.config)

    os.environ["CUDA_VISIBLE_DEVICES"] =  ""+str(args.gpu)

    ex.run()


#  python test_epic.py  --data_dir '/users/czhang/work/EK100_256p' --meta_dir '../data/epic_kitchens' --num_queries 12 --num_frames 16  --resume /users/czhang/work/EgoVLP/log-post-ddl/detr+egonce_bs128_lr3e-05_nq12_morepos_avglog_wordcontrast_vnmask_simmask06_strictmask_0_newboxes_resumebest_check/model/candidate1.pth.tar
