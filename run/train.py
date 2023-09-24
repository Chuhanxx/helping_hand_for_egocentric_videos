"""train DETR-like loss on detic results"""
import os
import numpy as np
from regex import D, P
import torch
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import torch.cuda.amp as amp 
from einops import rearrange 
import argparse
import logging
import sys
from collections import OrderedDict

sys.path.insert(0, "../")
import utils.tensorboard_utils as TB
from utils.train_utils import optim_policy, _valid_all_gather, verbose, format_nested_metrics_for_writer, set_path, AverageMeter, ProgressMeter, save_runtime_checkpoint
from model.loss import EgoNCE, WordContrastiveLoss
from model.metric import sim_matrix
from model.LaviLa import CLIP_OPENAI_TIMESFORMER_LARGE
from model.metric import egomcq_accuracy_metrics, compute_tv_accuracy
from model.tfm_decoder import ObjDecoder, Cross_Attention
from model.box_utils import build_matcher, SetCriterion, compute_box_loss

from data_loader.data_loader import MultiDistTextVideoDataLoader
from model.tokenizer import SimpleTokenizer


class AllGather_multi(torch.autograd.Function):
    """An autograd function that performs allgather on a tensor."""

    @staticmethod
    def forward(ctx, tensor, n_gpu, args):
        output = [torch.empty_like(tensor) for _ in range(args.world_size)]
        torch.distributed.all_gather(output, tensor)
        ctx.rank = args.rank
        ctx.batch_size = tensor.shape[0]
        return torch.cat(output, 0)

    @staticmethod
    def backward(ctx, grad_output):
        return (
            grad_output[ctx.batch_size * ctx.rank : ctx.batch_size * (ctx.rank + 1)],
            None, None,
        )


def prepare_data(data, device, tokenizer):
    # Aggregate and preprocess data input in one batch
    data['egoclip_text'] = data['text']
    data['text'] =  sum(data['rephrased_text'],[])
    if 'video_neg' in data.keys():  
        # with negative sampling
        if set(sum(data['rephrased_text'],[])) != set(['']): # if the rephrased texts list is not empty
            data['text_neg'] =  sum(data['rephrased_text_neg'],[])
        data['egoclip_text_neg'] = data['text_neg']
        data['text'] = data['text'] + data['text_neg']
        data['video'] = torch.cat( (data['video'], data['video_neg']), axis = 0).to(device)
        data['noun_vec'] = torch.cat((data['noun_vec'], data['noun_vec_neg']), axis=0).to(device)
        data['verb_vec'] = torch.cat((data['verb_vec'], data['verb_vec_neg']), axis=0).to(device)
        data['all_image_size'] = torch.cat((data['image_size'], data['image_size_neg']), axis=0).to(device)
        data['boxes'] = torch.cat((data['boxes'], data['boxes_neg']), axis=0)
        data['egoclip_text'] = tokenizer(data['egoclip_text']+data['egoclip_text_neg'])
    else:
        # without negative sampling
        data['video'] = data['video'].to(device)
        data['noun_vec'] = data['noun_vec'].to(device)
        data['verb_vec'] =  data['verb_vec'].to(device)
        data['all_image_size'] = data['image_size'].to(device)
        data['egoclip_text'] = tokenizer(data['egoclip_text'])
    data['noun_vec'][:,[102,504,364,321,556]] = 0 # remove hand/person and background nouns like floor, ground from nouns list
    data['text_str'] = data['text']
    data['text'] = tokenizer(data['text'])
    return data


def train_and_eval(loader,val_loader, model, backbone, tokenizer, optimizer, grad_scaler, device, epoch, args, best_acc):
    batch_time = AverageMeter('Time',':.2f')
    data_time = AverageMeter('Data',':.2f')
    losses = AverageMeter('Loss',':.4f')
    progress = ProgressMeter(
        len(loader), 
        [batch_time, data_time, losses],
        prefix='Epoch:[{}]'.format(epoch))

    # freeze backbone
    backbone.eval()
    end = time.time()
    tic = time.time()
    optimizer.zero_grad()

    if args.world_size == 1:  # identity function if not in distrbuted mode
        allgather = lambda x, *args, **kwargs: x
    else:
        allgather = AllGather_multi.apply

    all_nouns = torch.load(os.path.join(args.meta_dir,'noun_dict_lavila_embeds.pth'))
    for data_idx, data in enumerate(loader):
        data_time.update(time.time() - end)
        # ======================== Aggregate Input =====================================
        data = prepare_data(data, device, tokenizer)

        hand_boxes = data['boxes'][:,:,:2,:].to(device)
        obj_boxes = data['boxes'][:,:,2:,:].to(device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            with torch.no_grad():
                out = backbone(data['video'], data['text'].to(device),return_feature_map=True,  use_checkpoint=True)
            
            video_feature_map = out['image_feature_map'] # before projection 
            text_feature_map = out['text_feature_map'] # before projection 
            text_embeds = out['text_embed'] # after projection 
            grid_side = int(((video_feature_map.shape[1]-1)/4)**0.5)
            video_grid = rearrange(video_feature_map[:, 1:, :], 'b (t h w) c -> b t (h w) c', t=data['video'].shape[1], h=grid_side, w=grid_side)

            B = data['video'].shape[0]
            model_out, hs, _, _ = model(video_grid,use_checkpoint=False)
            sentence_len =  data['text'].argmax(dim=-1) 
            N_rephrased = sentence_len.shape[0]
    
            # Freature projection 
            text_embeds = model.txt_proj(text_feature_map[torch.arange(N_rephrased),sentence_len,:])
            video_embeds = model.obj_proj(hs[-1, :])[:,-1,...]
            video_embeds = allgather(video_embeds.contiguous(), args.world_size, args)
            text_embeds = allgather(text_embeds.contiguous(), args.world_size, args)
            text_tokens = allgather(data['text'].to(device), args.world_size, args)

            # Initialize the loss dictionary.
            loss_dict = {}
            loss_dict['total_loss'] = 0

            # ======================== Modified EgoNCE Loss ================================
            text_embeds = allgather(text_embeds, args.world_size, args)
            video_level_similarity = sim_matrix(text_embeds, video_embeds)
            # Follow the section-aware Positive Sampling in EgoVLP to mask out samples with high similarity 
            # where the simuilarity is based on verb & noun similarity.
            verb_vec = allgather(data['verb_vec'], args.world_size, args)
            noun_vec = allgather(data['noun_vec'], args.world_size, args)
            sim_v = sim_matrix(verb_vec, verb_vec)
            sim_n = sim_matrix(noun_vec, noun_vec)        
            # pad mask for rephrased texts, 1 for non-padded , 0 for padded.
            rephrased_pad_mask = ((text_tokens !=0).sum(-1)!=2).float()[:,None]
            rephrased_pad_mask = rephrased_pad_mask.repeat(1,video_embeds.shape[0])
            nce_loss, _ = EgoNCE()(video_level_similarity, sim_v, sim_n, 
                                multi_pad_mask=rephrased_pad_mask, 
                                strict_mask=True)  
            loss_dict['total_loss'] += nce_loss
            loss_dict.update({
                        'nce-loss': nce_loss.detach()})
            # compute accuracy
            # Use the first text embeds to compute acc.
            video_level_similarity = video_level_similarity.view(B*args.world_size,
                                                                 -1,
                                                                 B*args.world_size)[:,0,:] 
            acc_vt, acc_tv = compute_tv_accuracy(video_level_similarity, text_embeds, sim_v, sim_n, B*args.world_size, device)
      
            # ======================== BBox Loss ================================
            box_loss = 0
            hand_boxes = hand_boxes.flatten(0,1)
            obj_boxes = obj_boxes.flatten(0,1) 
            all_image_size = data['all_image_size'][:,None,:].expand(-1,4,-1).flatten(0,1)
            n_q = args.num_queries if args.num_queries !=0 else 10
            # loss for hand boxes
            box_loss_hand, _ = compute_box_loss('hand_boxes',
                                                args.criterion,
                                                model_out,
                                                hand_boxes,
                                                None,
                                                all_image_size,
                                                n_queries=n_q)
            box_loss += box_loss_hand
            # loss for object boxes
            box_loss_obj, _ = compute_box_loss('obj_boxes',
                                                args.criterion,
                                                model_out,
                                                obj_boxes,
                                                None,
                                                all_image_size,
                                                n_queries=n_q)
            box_loss += box_loss_obj
            loss_dict['total_loss'] += box_loss
            loss_dict['box_loss']  = torch.tensor(box_loss)

            # ======================== Word Contrastive Loss ================================
            noun_embeds = model.txt_proj(torch.stack([*all_nouns.values()]).to(device))
            noun_gt_inds = torch.cat([data['nouns'], data['nouns_neg']]).to(device).to(torch.int64)
            pred_noun_embeds = model.obj_proj(hs[-1, :])[:,:-1,...]
            word_loss = WordContrastiveLoss()(noun_embeds, pred_noun_embeds, noun_gt_inds)
            loss_dict['total_loss'] += 0.5*word_loss
            loss_dict['word-nce-loss'] = word_loss.detach()

            loss_dict.update({
                'top1-video-to-text': acc_vt,
                'top1-text-to-video': acc_tv,})

        # backward
        grad_scaler.scale(loss_dict['total_loss']).backward()
        grad_scaler.unscale_(optimizer)
        grad_scaler.step(optimizer)
        grad_scaler.update()
        optimizer.zero_grad()
        print(f'backward time {time.time()-tic}')
        tic = time.time()

        if data_idx == 0:
            avg_meters = {k: AverageMeter(f'{k}:',':.4f') for k in loss_dict.keys()}
        for metric, value in loss_dict.items():
            avg_meters[metric].update(value.item(), B)

        batch_time.update(time.time() - end)
        progress.display(args.iteration%len(loader))
        print('\t' + ' '.join([f"{k}:{v.item():.3f}" for k,v in loss_dict.items()]))

        if args.iteration % 5 == 0:
            for k, v in loss_dict.items():
                args.train_plotter.add_data(f'local/{k}', v.item(), args.iteration)
            args.train_plotter.add_data('device/sps', 1/(time.time()-end), args.iteration)
            args.train_plotter.log_gpustat(step=args.iteration)

        end = time.time()
        args.iteration += 1

        if args.iteration % args.runtime_save_iter == 0:
            print('saving runtime checkpoint ...')
            if hasattr(model, 'module'):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            save_dict = {
                'epoch': epoch,
                'state_dict': state_dict,
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
                'iteration': args.iteration,}
            save_runtime_checkpoint(save_dict, 
                filename=os.path.join(args.model_path, 'runtime.pth.tar'), 
                rm_history=True)

        if args.iteration % args.eval_freq ==0:
            val_loss = evaluate(val_loader, model, backbone, tokenizer, device, epoch, args)
            if args.rank == 0:
                is_best = val_loss['t2i_acc']['Inter-video'] > best_acc  # temporary use val loss
                best_acc = max(val_loss['t2i_acc']['Inter-video'], best_acc)
                if hasattr(model, 'module'):
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()     

                save_dict = {
                    'epoch': epoch,
                    'state_dict': state_dict,
                    'best_acc': best_acc,
                    'optimizer': optimizer.state_dict(),
                    'iteration': args.iteration}

                if is_best:
                    save_runtime_checkpoint(save_dict, 
                        filename=os.path.join(args.model_path, 'best.pth.tar'), rm_history=False)
            model.train() # reset it back to train for training 



    print(f'epoch {epoch} finished, takes {time.time() - tic} seconds')
    for metric_name, avg_meter in avg_meters.items():
        args.train_plotter.add_data(f'global/{metric_name}', avg_meter.avg, epoch)

    return val_loss, best_acc



@torch.no_grad()
def evaluate(loader, model, backbone, tokenizer, device, epoch, args,):
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
        dl_idx = 0
        for batch_idx, data in enumerate(tqdm(loader)):

            # short cycle for online evaluation
            if batch_idx > 1000:
                break
            
            data['video'] = data['video'][0]  # remove batch
            data['text_str'] = data['text']
            data['video'] = data['video'].to(device)
            data['text'] = tokenizer(data['text']).unsqueeze(0)

            out = backbone(data['video'], data['text'].to(device),return_feature_map=True)
            video_feature_map = out['image_feature_map'] # before projection 
            text_feature_map = out['text_feature_map'] # before projection 

            # first get the video and text features from the backbone
            grid_side = int(((video_feature_map.shape[1])/4)**0.5)
            video_grid = rearrange(video_feature_map[:, 1:, :], 'b (t h w) c -> b t (h w) c', t=4, h=grid_side, w=grid_side)
            sen_lens =  data['text'].argmax(dim=-1)
            text_embeds =  model.txt_proj(text_feature_map[0,sen_lens])
            model_out, hs,_, _ = model(video_grid)

            video_embeds = model.obj_proj(hs[-1, :])[:,-1,...]
  
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
    t2i_arr_cat = torch.cat(t2i_arr[dl_idx])
    type_cat = torch.cat(type_arr[dl_idx])

    metric_name = 'egomcq_accuracy_metrics'
    res_t2i = egomcq_accuracy_metrics(t2i_arr_cat, gt_arr_cat, type_cat)

    torch.save(save_dict, f'{args.log_path}/{epoch}-{args.iteration}_results_{args.results_suffix}.pth')
    if args.rank == 0:
        logging.info(
            verbose(epoch=epoch, iter=args.iteration, metrics=res_t2i, name='ti2'))
    nested_metrics[dl_idx][metric_name+'_t2i'] = res_t2i

    if args.val_plotter is not None and args.rank == 0:
        to_write_t2i = format_nested_metrics_for_writer(res_t2i, mode=metric_name,name='egomcq'+ '_t2i')
        for key, val in to_write_t2i.items():
            key = key.replace('[', '_').replace(']', '_')
            args.val_plotter.add_data(f'Val_metrics_t2i_{dl_idx}/{key}', val, args.iteration)
    res_dict = {}

    if args.rank == 0:
        res_dict = {f'val_loss_{dl_idx}': total_val_loss[dl_idx] / len(gt_arr[dl_idx])}
        res_dict['nested_val_metrics'] = nested_metrics
        res_dict['t2i_acc'] = res_t2i

    return res_dict


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
        args.batch_size = num_gpu * args.batch_size
        print('=> Effective BatchSize = %d' % args.batch_size)
    else:
        args.num_gpu = 0
        device = torch.device('cpu')
        print('=> Run with CPU')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    args.log_path, args.model_path, args.exp_path = set_path(args)
    writer_train = SummaryWriter(logdir=os.path.join(args.log_path, 'train'),
                                flush_secs=60)
    args.train_plotter = TB.PlotterThread(writer_train)
    writer_val = SummaryWriter(logdir=os.path.join(args.log_path, 'val'),
                            flush_secs=60)
    args.val_plotter = TB.PlotterThread(writer_val)
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
                text_use_cls_token=False,
                project_embed_dim=256,
                num_frames=4,
                temperature_init=0.07,
            )
    feature_dim = 1024
    checkpoint = torch.load(os.path.join(args.meta_dir,'clip_openai_timesformer_large.narrator_rephraser.ep_0003.md5sum_c89337.pth'), map_location='cpu')
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
    
    num_queries = args.num_queries+1

    tfm = Cross_Attention(normalize_before=True,
                    return_intermediate_dec=True,)
    model = ObjDecoder(transformer=tfm, 
                    num_classes=22047, # not used 
                    num_queries=num_queries, 
                    aux_loss=True,
                    pred_traj=True,
                    feature_dim=feature_dim,
                    self_attn=False)

    args.matcher = build_matcher(args)
    weight_dict = {"loss_bbox_hand_boxes": 5, 
                   "loss_bbox_obj_boxes": 5 ,
                   "loss_giou_hand_boxes": 2,
                   "loss_giou_obj_boxes": 2,}

    args.loss = [i.lower() for i in args.loss]
    args.criterion = SetCriterion(
            22047, # not used 
            matcher=args.matcher, 
            weight_dict=weight_dict, 
            eos_coef=0.1, 
            losses=["boxes", "cardinality"],
        )
    args.criterion.to(device)
    model.to(device)
    model_without_dp = model

    text_params={"input": "text"}
    video_params={
        "input_res": 224,
        "num_frames": args.num_frames,
        "loading": "lax"
    }

    train_loader = MultiDistTextVideoDataLoader(args,
                                        'EgoClip',
                                        text_params,
                                        video_params,
                                        args.data_dir,
                                        meta_dir=args.meta_dir,
                                        split='train',
                                        batch_size=args.batch_size,                        
                                        num_workers=args.num_workers,
                                        video_res=args.video_res,
                                        reader='cv2_egoclip',
                                        shuffle=True, 
                                        neg_param=False,
                                        tsfm_params= tsfm_params,
                                        )

    val_loader = MultiDistTextVideoDataLoader(args,
                                        'EgoClip',
                                        text_params,
                                        video_params,
                                        args.data_dir,
                                        meta_dir=args.meta_dir,
                                        split='val',
                                        batch_size=1,  
                                        video_res=args.video_res,
                                        num_workers=args.num_workers,
                                        reader='cv2_egoclip',
                                        shuffle=False,
                                        subsample='mcq',
                                        neg_param=False,
                                        tsfm_params= tsfm_params,
                                        )

    backbone.to(device)
    ### optimizer ###
    params = optim_policy(backbone, model, args.lr, args.wd)
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    best_acc = 0
    ### restart ###
    if args.resume:
        print(f"resume from checkpoint {args.resume}")
        args.resume = get_model_card(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        state_dict = checkpoint['state_dict']
        args.start_epoch = checkpoint['epoch']
        args.iteration = checkpoint['iteration']
        best_acc = checkpoint['best_acc']
        try:
            model_without_dp.load_state_dict(state_dict)
        except:
            missing_keys, unexpected_keys = model_without_dp.load_state_dict(state_dict, strict=True)
            if len(missing_keys):
                print(f'[Missing keys]:{"="*12}\n{chr(10).join(missing_keys)}\n{"="*20}')
            if len(unexpected_keys):
                print(f'[Unexpected keys]:{"="*12}\n{chr(10).join(unexpected_keys)}\n{"="*20}')
            user_input = input('[WARNING] Non-Equal load for resuming training, continue? [y/n]')
            if user_input.lower() == 'n':
                sys.exit()
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f'Not resuming optimizer states due to Error: {e}\nInitialized the optimizer instead...')


    args.decay_steps = args.epochs * len(train_loader) 
    args.warmup_epochs = float(args.epochs / 20)
    grad_scaler = amp.GradScaler()
    tokenizer = SimpleTokenizer()
            
    for epoch in range(args.start_epoch, args.epochs):
        np.random.seed(args.iteration//2000)
        random.seed(args.iteration//2000)
        torch.manual_seed(args.iteration//2000)

        val_loss,best_acc = train_and_eval(train_loader, val_loader, model, backbone, tokenizer, optimizer, grad_scaler, device, epoch, args, best_acc)
        # eval after one epoch finishes
        is_best = val_loss['t2i_acc']['Inter-video'] > best_acc  # temporary use val loss
        best_acc = max(val_loss['t2i_acc']['Inter-video'] , best_acc)
        state_dict = model_without_dp.state_dict()
        save_dict = {
            'epoch': epoch,
            'state_dict': state_dict,
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'iteration': args.iteration}
        save_runtime_checkpoint(save_dict, 
            filename=os.path.join(args.model_path, 
            'best.pth.tar', 
            rm_history=False))

    print('Training from ep %d to ep %d finished' % (args.start_epoch, args.epochs))
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
    parser.add_argument('--seed', default=111,type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=3e-5, type=float)
    parser.add_argument("--loss", nargs="+", default=["a", "b"])
    parser.add_argument('--wd', default=1e-5, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--name_prefix', default='', type=str)
    parser.add_argument('--backbone', default='EgoVLP', type=str)
    parser.add_argument('--results_suffix', default='', type=str)
    parser.add_argument('--meta_dir', default='../data/EgoClip', type=str)
    parser.add_argument('--data_dir', default='./', type=str)
    parser.add_argument('--num_frames', default=4, type=int)
    parser.add_argument('--eval_freq', default=2500, type=int)
    parser.add_argument('--video_res', default=224, type=int)
    parser.add_argument('--runtime_save_iter', default=2500, type=int)
    parser.add_argument('--optim', default='adamw', type=str)
    parser.add_argument('--num_queries', default=12, type=int)
    parser.add_argument('--raw_resolution', default=256, type=int)

    parser.add_argument('-k', '--local_rank', type=int, default=local_rank)
    parser.add_argument('-ws', '--world_size', type=int, default=world_size)
    parser.add_argument('-rk', '--rank', type=int, default=rank)
    parser.add_argument('-j', '--num_workers', default=8, type=int)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)


