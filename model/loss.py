import torch
import torch.nn.functional as F
from torch import nn
from model.metric import sim_matrix
from scipy.optimize import linear_sum_assignment


class EgoNCE(nn.Module):
    def __init__(self, temperature=0.07, noun=True, verb=True):
        super().__init__()
        self.noun = noun
        self.verb = verb
        self.temperature = temperature

    def forward(self, x, mask_v, mask_n, multi_pad_mask=None, strict_mask=False, vn_threshold=0):
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

        mask_bool = mask > vn_threshold

        "Assumes input x is similarity matrix of N x M \in [-1, 1], computed using the cosine similarity between normalised vectors"
        i_sm = masked_x/self.temperature
        j_sm = masked_x.t()/self.temperature
        idiag = torch.sum(torch.log_softmax(i_sm, dim=1) * mask_bool, dim=1)
        idiag = idiag / mask_bool.sum(-1)
        loss_i = idiag.sum() / len(idiag)

        jdiag = torch.sum(torch.log_softmax(j_sm, dim=1) * mask_bool.t(), dim=1)
        jdiag = jdiag  / mask_bool.sum(0)
        loss_j = jdiag.sum() / len(jdiag)
        return - loss_i - loss_j, mask_bool

class WordContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, noun_threshold=0.6):
        super().__init__()
        self.temperature = temperature
        self.noun_threshold = noun_threshold

    def forward(self, noun_embeds, pred_noun_embeds, noun_gt_inds):
        gt_noun_embeds = noun_embeds.index_select(0, noun_gt_inds.flatten())
        gt_noun_embeds = gt_noun_embeds.view(noun_gt_inds.shape[0], noun_gt_inds.shape[1], -1)

        # match the predicted objects with gt nouns
        word_sim = -sim_matrix(gt_noun_embeds,pred_noun_embeds)
        n_words_per_sample = (noun_gt_inds!=0).float().sum(-1)
        n_words_per_sample = [int(i.item()) for i in n_words_per_sample]
        word_sim = word_sim.flatten(0,1)[noun_gt_inds.flatten()!=0]
        cost_matrix = word_sim.detach().cpu().split(n_words_per_sample,0)
        selected_output = []

        for i, c in enumerate(cost_matrix):
            if c.shape[0] !=0:
                row_ind, col_ind = linear_sum_assignment(c)
                selected_output.append(pred_noun_embeds[i][col_ind])
        selected_output = torch.cat(selected_output)
        # compute the similarity between selected predictions with all the nouns
        sim_all = sim_matrix(selected_output,noun_embeds)
        # ignore nouns with simialr meanings, ignore all the nouns with similarity > noun_threshold
        noun_sim = sim_matrix(noun_embeds,noun_embeds)
        N = noun_sim.shape[0]
        noun_sim[torch.arange(N),torch.arange(N)] = 0
        noun_mask = noun_sim.index_select(0,noun_gt_inds[noun_gt_inds!=0])>self.noun_threshold

        word_loss = torch.nn.functional.cross_entropy(
            sim_all.masked_fill(noun_mask,-1)/self.temperature, 
            noun_gt_inds[noun_gt_inds!=0]) 
        return word_loss
