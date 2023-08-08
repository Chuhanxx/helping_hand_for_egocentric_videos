import pdb
import torch
import torch.nn.functional as F
from torch import nn
import pickle
from torch import einsum
from einops import rearrange, repeat, reduce
import matplotlib.pyplot as plt 
import numpy as np
from torch.autograd import Variable
from model.metric import sim_matrix

class NormSoftmaxLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()

        self.temperature = temperature

    def forward(self, x):
        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_logsm = F.log_softmax(x/self.temperature, dim=1)
        j_logsm = F.log_softmax(x.t()/self.temperature, dim=1)

        # sum over positives
        idiag = torch.diag(i_logsm)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.diag(j_logsm)
        loss_j = jdiag.sum() / len(jdiag)

        return - loss_i - loss_j

class EgoNCE(nn.Module):
    def __init__(self, temperature=0.07, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n, multi_pad_mask=None, logavg=True, strict_mask=False, vn_threshold =0.6):
        # mask_diag = torch.eye(x.shape[0]).cuda()
        # print((multi_pad_mask.sum(-1)!=0).sum())
        if multi_pad_mask is None: # single positive sample 
            mask_diag = torch.eye(x.shape[0], device=x.device)

            if mask_v is not None and mask_n is not None:
                mask = mask_v * mask_n + mask_diag
            elif mask_n is not None:
                mask = mask_n + mask_diag
            elif mask_v is not None:
                mask = mask_v + mask_diag
            masked_x = x
        else: # multiple positive sample 
            masked_x = x.masked_fill(~multi_pad_mask.bool(),float('-inf'))

            # create diagonal mask for postive text-video pairs
            multi_pos_mask = torch.eye(x.shape[-1], device=x.device)[:,None,:]
            R = multi_pad_mask.shape[0]//multi_pad_mask.shape[1] # how many captions per video
            multi_pos_mask = multi_pos_mask.repeat(1,R,1).flatten(0,1)
          
            # add more positives to the mask -- sentences that share the same verb and nouns as positives
            if mask_v is not None and mask_n is not None:
                mask_v = mask_v[:,None,:].repeat(1,R,1).flatten(0,1)
                mask_n= mask_n[:,None,:].repeat(1,R,1).flatten(0,1)
                
                mask_vn =  (mask_v * mask_n) * multi_pad_mask
                mask_pos = (multi_pos_mask) * multi_pad_mask
                mask =  (mask_v * mask_n + multi_pos_mask) * multi_pad_mask

                mask_vn = mask_vn[masked_x.sum(-1)!=float('-inf')]
                mask_pos =  mask_pos[masked_x.sum(-1)!=float('-inf')]
                mask =  mask[masked_x.sum(-1)!=float('-inf')]

            elif mask_n is not None:
                mask_n= mask_n[:,None,:].repeat(1,5,1).flatten(0,1)
                mask = (mask_n + multi_pos_mask ) * multi_pad_mask

                mask = mask[masked_x.sum(-1)!=float('-inf')]
            elif mask_v is not None:
                mask_v = mask_v[:,None,:].repeat(1,5,1).flatten(0,1)
                mask = (mask_v + multi_pos_mask ) * multi_pad_mask
                mask = mask[masked_x.sum(-1)!=float('-inf')]

            masked_x = masked_x[masked_x.sum(-1)!=float('-inf')]

        if strict_mask:
            mask_bool = mask > vn_threshold
            mask_vn_bool = mask_vn > vn_threshold
            mask_pos_bool = mask_pos >0 
        else:
            mask_bool = mask > 0
            mask_vn_bool = mask_vn > 0
            mask_pos_bool = mask_pos > 0 

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        if logavg:
            i_sm = F.softmax(masked_x/self.temperature, dim=1)
            j_sm = F.softmax(masked_x.t()/self.temperature, dim=1)
            
            i_sum = torch.sum(i_sm * mask_vn_bool, dim=1) 
            nonzero_mask_i = (i_sum!=0).float() 
            vn_pos_i = torch.log(i_sum+1e-9)
            # vn_pos_i[vn_pos_i==float('-inf')] = 0
            loss_i = (vn_pos_i * nonzero_mask_i).sum() / nonzero_mask_i.sum()

            j_sum = torch.sum(j_sm * mask_vn_bool.t(), dim=1)
            nonzero_mask_j = (j_sum!=0).float() 
            vn_pos_j = torch.log(j_sum  + 1e-9)
            mutli_pos = torch.sum(torch.log(j_sm) * mask_pos_bool.t(), dim=1)
            jdiag = (vn_pos_j*nonzero_mask_j + mutli_pos) / (mask_pos_bool.sum(0)+ nonzero_mask_j )
            loss_j = jdiag.sum() / len(jdiag)
            # mutli_pos = torch.sum(torch.log(i_sm) * mask_pos_bool, dim=1)
      
            # idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))

            # idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1))
            # loss_i = idiag.sum() / len(idiag)

            # jdiag = torch.log(torch.sum(j_sm * mask_bool.t(), dim=1))
            # loss_j = jdiag.sum() / len(jdiag)
        else: # avg(log P)
            i_sm = masked_x/self.temperature
            j_sm = masked_x.t()/self.temperature
            idiag = torch.sum(torch.log_softmax(i_sm, dim=1) * mask_bool, dim=1)
            idiag = idiag  / mask_bool.sum(-1)
            loss_i = idiag.sum() / len(idiag)

            jdiag = torch.sum(torch.log_softmax(j_sm, dim=1) * mask_bool.t(), dim=1)
            jdiag = jdiag  / mask_bool.sum(0)
            loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j, mask_bool

class MaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.2, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin = F.relu(self.margin - (x1_ - x2_))

        return max_margin.mean()

class AdaptiveMaxMarginRankingLoss(nn.Module):

    def __init__(self, margin=0.4, fix_norm=True):
        super().__init__()
        self.fix_norm = fix_norm
        self.loss = nn.MarginRankingLoss(margin)
        self.margin = margin

    def forward(self, x, weight=None):
        n = x.size()[0]

        x1 = torch.diag(x)
        x1 = x1.unsqueeze(1)
        x1 = x1.expand(n, n)
        x1 = x1.contiguous().view(-1, 1)
        x1 = torch.cat((x1, x1), 0)

        w1 = weight.unsqueeze(1)
        w1 = w1.expand(n, n)
        w1 = w1.contiguous().view(-1, 1)
        w1 = torch.cat((w1, w1), 0)

        x2 = x.view(-1, 1)
        x3 = x.transpose(0, 1).contiguous().view(-1, 1)

        x2 = torch.cat((x2, x3), 0)
        max_margin = F.relu(  w1 * self.margin - (x1 - x2))

        if self.fix_norm:
            # remove the elements from the diagonal
            keep = torch.ones(x.shape) - torch.eye(x.shape[0])  # 128 x 128
            keep1 = keep.view(-1, 1)
            keep2 = keep.transpose(0, 1).contiguous().view(-1, 1)
            keep_idx = torch.nonzero(torch.cat((keep1, keep2), 0).flatten()).flatten()
            if x1.is_cuda:
                keep_idx = keep_idx.cuda()
            x1_ = torch.index_select(x1, dim=0, index=keep_idx)
            w1_ = torch.index_select(w1, dim=0, index=keep_idx)
            x2_ = torch.index_select(x2, dim=0, index=keep_idx)
            max_margin =  F.relu( w1_ * self.margin - (x1_ - x2_))

        return max_margin.mean()

class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.loss(output, target)


def masked_mean(t, mask, dim = 1, eps = 1e-6):
    t = t.masked_fill(~mask, 0.)
    numer = t.sum(dim = dim)
    denom = mask.sum(dim = dim).clamp(min = eps)
    return numer / denom

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def max_neg_value(dtype):
    return -torch.finfo(dtype).max

def matrix_diag(t):
    device = t.device
    i, j = t.shape[-2:]
    num_diag_el = min(i, j)
    i_range = torch.arange(i, device = device)
    j_range = torch.arange(j, device = device)
    diag_mask = rearrange(i_range, 'i -> i 1') == rearrange(j_range, 'j -> 1 j')
    diag_el = t.masked_select(diag_mask)
    return rearrange(diag_el, '(b d) -> b d', d = num_diag_el)


class DenseCLIP(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_embeds, video_embeds, text_attn_mask=None, 
                    return_sim=False, has_cls_token=True, only_return_sim=False):
        """
        text_embeds: B, 1 + seqlen, C.
        video_embeds: B, 1 + grid*grid, C.
        text_attn_mask: B, 1 + seqlen. 1 means to attend. 0 means ignore (pad).
        """
        if has_cls_token:
            trim = 1 
        else:
            trim = 0
        # normalize first, to compute cosine distance
        text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)
        video_embeds = video_embeds / video_embeds.norm(dim=1, keepdim=True)

        if text_attn_mask is None:
            text_attn_mask = torch.ones_like(text_embeds[:,:,0])

        # remove CLS on both streams
        text_attn_mask = text_attn_mask[..., trim:].bool().squeeze(1)
        text_latents = text_embeds[:, trim:, :]
        # visual_latents = video_embeds[:, trim:, :]
        visual_latents = video_embeds

        # below is modified from Phil Wang: https://github.com/lucidrains/x-clip/blob/main/x_clip/x_clip.py#L720
        sim_text_to_image = einsum('x t d, y i d -> x y t i', text_latents, visual_latents) * self.temperature
        sim_image_to_text = sim_text_to_image
        
        # for i in range(sim_text_to_image.shape[1]):
        #     plt.subplot(1, sim_text_to_image.shape[1], i+1)
        #     plt.imshow(sim_text_to_image[0,i][:text_attn_mask[0].sum()].detach().cpu(),
        #         cmap=plt.cm.jet,
        #         interpolation=None)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('text_to_image.png')
        # import ipdb; ipdb.set_trace()
        
        if only_return_sim:
            return sim_text_to_image, sim_image_to_text

        text_to_image = reduce(sim_text_to_image, '... t i -> ... t', 'max')
        text_to_image_mask = rearrange(text_attn_mask, 'b t -> 1 b 1 t')
        text_to_image = masked_mean(text_to_image, text_to_image_mask, dim = -1)
     

        image_to_text_mask = rearrange(text_attn_mask, 'b t -> 1 b 1 t 1')
        masked_sim = sim_image_to_text.masked_fill(~image_to_text_mask, max_neg_value(sim_image_to_text.dtype))
        image_to_text = reduce(reduce(masked_sim, '... t i -> ... i', 'max'), '... i -> ...', 'mean')



        # text_to_image = rearrange(text_to_image, 'm n ... -> (m n) ...')
        # image_to_text = rearrange(image_to_text, 'm n ... -> (m n) ...')

        # exponentiate
        text_to_image_exp, image_to_text_exp = map(torch.exp, (text_to_image, image_to_text))

        # numerators
        text_to_image_pos, image_to_text_pos = map(matrix_diag, (text_to_image_exp, image_to_text_exp))

        # denominator
        text_to_image_denom, image_to_text_denom = map(lambda t: t.sum(dim = -1), (text_to_image_exp, image_to_text_exp))

        # loss
        text_to_image_loss = -log(text_to_image_pos / text_to_image_denom).mean(dim = -1)
        image_to_text_loss = -log(image_to_text_pos / image_to_text_denom).mean(dim = -1)

        # calculate CL loss
        # cl_losses = (text_to_image_loss + image_to_text_loss) / 2
        cl_losses = text_to_image_loss
        # final
        loss = cl_losses

        if return_sim:
            return (loss, text_to_image[0,:], image_to_text[0,:])
        else:
            return loss


class ExtractVisualCLS(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs

class ExtractDense(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs

class ExtractMeta(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs

class EvalDETR(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs

class EvalDeticConcat(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, inputs):
        return inputs


class EgoNCEWithFeature(nn.Module):
    def __init__(self, temperature=0.07, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n):
        mask_diag = torch.eye(x.shape[0]).cuda()
        if self.noun and self.verb:
            mask = mask_v * mask_n + mask_diag
        elif self.noun:
            mask = mask_n + mask_diag
        else:
            mask = mask_v + mask_diag

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)

        mask_bool = mask > 0
        idiag = torch.log(torch.sum(i_sm * mask_bool, dim=1) )
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j



class WordContrast(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, x, pos_mask):
        # mask_diag = torch.eye(x.shape[0]).cuda()
        i_sm = F.softmax(x/self.temperature, dim=1)
        j_sm = F.softmax(x.t()/self.temperature, dim=1)
        mask_bool = pos_mask.bool() 

        idiag = torch.log(torch.sum(i_sm * mask_bool , dim=1) )
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.log(torch.sum(j_sm * mask_bool, dim=1) )
        loss_j = jdiag.sum() / len(jdiag)

        i_argmax_onehot = torch.zeros_like(i_sm).scatter_(0, torch.argmax(i_sm,-1).unsqueeze(0), 1.)
        acc_i = ((i_argmax_onehot * mask_bool).sum(0) > 0).float().mean()

        j_argmax_onehot = torch.zeros_like(j_sm).scatter_(-1, torch.argmax(j_sm,-1).unsqueeze(-1), 1.)
        acc_j = ((j_argmax_onehot * mask_bool).sum(-1) > 0).float().mean()

        return (acc_i+acc_j)/2, - loss_i - loss_j
