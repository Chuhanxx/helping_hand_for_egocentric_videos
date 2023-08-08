import os
import sys
import json
import pandas as pd
sys.path.insert(0, "../")

import torch
import pickle
from PIL import Image

from torchvision import transforms
from torch.utils.data import Dataset
from ast import literal_eval


from base.base_dataset import TextVideoDataset
from data_loader.transforms import  init_video_transform_dict,custom_img_crop
from utils.box_ops import load_hand_boxes, crop_boxes

from utils.utils import  img_denorm 

class EgoClip_EgoMCQ(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'egoclip.csv',
            'val': 'egomcq.json',
            'test': 'egomcq.json'
        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary
        self.handobj_dir = os.path.join(self.data_dir,'hand_object_clip_per_video_4f_lavila_narrator_640')
        path_narration_noun = os.path.join(self.meta_dir,'narration_noun_taxonomy.csv')
        self.rephrased_txts =  torch.load(os.path.join(self.meta_dir,'lavila_rephrased.pth'))

        self.noun_pd = pd.read_csv(path_narration_noun,converters={"group": literal_eval})
        self.noun_dict = torch.load(os.path.join(self.meta_dir,'noun_dict_lavila_embeds.pth'))
        self.all_nouns = [*self.noun_dict.keys()]
        self.counter = 0
        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t',error_bad_lines=False)
            if isinstance(self.subsample,list):
                self.metadata = self.metadata.iloc[self.subsample[0]:self.subsample[1]]
            self.frame_sample = self.video_params.get('frame_sample', 'uniform')

            if self.neg_param and self.split == 'train':
                self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
                self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + (self.metadata['chunk_id']//self.chunk_sec).astype(str)
                self.metadata_by_segment_id = dict(tuple(self.metadata.groupby('segment_id')))

            self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)

        elif self.split in ['val', 'test']:
            self.frame_sample = 'uniform'
            with open(os.path.join(self.meta_dir, target_split_fp), 'r') as load_f:
                self.metadata = json.load(load_f)
            
            if self.split == 'val': # enable faster eval
                print('Eval on a subset for speed ...')

                sorted_keys = sorted(list(self.metadata.keys()), key=lambda x:int(x))
                # sample both inter and intra
                keys_by_type = {}
                for key, value in self.metadata.items():                
                    qtype = value['types']
                    if qtype not in keys_by_type:
                        keys_by_type[qtype] = []
                    keys_by_type[qtype].append(key)

                metadata_subset = set(sorted_keys[0:len(self.metadata)])

                sorted_keys_head = sorted(keys_by_type[1], key=lambda x:int(x))
                sorted_keys_tail = sorted(keys_by_type[2], key=lambda x:int(x))
                metadata_subset = set(sorted_keys_head).union(set(sorted_keys_tail))

                self.metadata = {k:v for k,v in self.metadata.items() if k in metadata_subset}
                self.metakeys = sorted(self.metadata.keys())


    def load_hand_object_box(self, sample):
        clip_start = float(sample['clip_start'])

        hand_boxes = torch.zeros(4,2,4)
        obj_boxes = torch.zeros(4,2,4)       
        image_size = (0, 0)


        video_name = sample['video_uid']
        clip_index = str(int(clip_start//self.chunk_sec))
        hand_file = os.path.join(self.handobj_dir,video_name,clip_index+'.handobj.pkl')
        if os.path.exists(hand_file): 
            hand_info = pickle.load(open(hand_file,'rb'))
            poss_starts = [clip_start,clip_start-0.001,clip_start+0.001]
            image_size = ([*hand_info.values()][0]['info']['height'],[*hand_info.values()][0]['info']['width'])
            for start in poss_starts:
                try:
                    hand_boxes = torch.stack([load_hand_boxes(hand_info[round(start,3)],i) for i in range(4)])
                    obj_boxes = torch.stack([load_hand_boxes(hand_info[round(start,3)],i,box_type='obj_dets') for i in range(4)])
                    scuess = 1
                except:
                    scuess = 0
                if scuess ==1:
                    break
        boxes = torch.cat([hand_boxes,obj_boxes],1)
        return boxes, image_size


    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_dir, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_dir, video_uid, str(chunk_end_id) + ".mp4")

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec

    def _get_video_frames(self, video_fp, video_sec, bound_sec, boxes=None, pred=False):
        video_loading = self.video_params.get('loading', 'strict')
        try:

            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, seconds = self.video_reader(video_fp[0], video_sec[0], end_second=video_sec[1], 
                                            clip_length=self.video_params['num_frames'])
                valid = 1 
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0).to(torch.float32)
                valid = 0
        if boxes is not None and boxes.sum()!=0:
            imgs,crop_params = custom_img_crop(imgs,boxes*256/self.video_res,pred=pred)
            crop_params = crop_params*self.video_res/256
        else:
            crop_params = torch.tensor([0.,0.,0.,0.])

        
        im_size = imgs.shape[2:]

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs

        return final,im_size,crop_params,valid

    def _get_caption(self, sample):
        noun_vec = torch.zeros(self.noun_dim)
        verb_vec = torch.zeros(self.verb_dim)
        noun_idx = eval(sample['tag_noun'])
        verb_idx = eval(sample['tag_verb'])
        for i in noun_idx:
            noun_vec[i] = 1
        for i in verb_idx:
            verb_vec[i] = 1

        return sample['clip_text'], noun_vec, verb_vec

    def _get_rephrased_caption(self, sample, video_sec, caption):
        segment_id = sample['video_uid']
        clip_id = str(int(video_sec[0] //600))
        cs = round(video_sec[0],1)

        rephrased_texts = ['']*5
        if segment_id in self.rephrased_txts:
            if clip_id in self.rephrased_txts[segment_id]:
                for s in torch.arange(cs-0.5,cs+0.5,0.1):
                    s = round(s.item(),1)
                    # rerephrased_text_path = os.path.join(self.text_dir, segment_id, clip_id, str(round(s.item(),1))+'.pth.tar' )
                    # rephrased = torch.load(rerephrased_text_path)
                    rephrased = self.rephrased_txts[segment_id][clip_id]
                    if s in rephrased:
                        for j,line in enumerate(rephrased[s]):
                            rephrased_texts[j] = line[0]

        # there are some misalignment between the start time in LaviLa and egoclip 
        # if the first caption cannot be matched, we choose not use the rephrased captions
        if rephrased_texts[0] != caption:
            rephrased_texts = ['']*5
            rephrased_texts[0] = caption

        return rephrased_texts
        


    def extract_noun(self,sample, caption):
        def p(word):
            word = word.replace('.','').replace(',','')
            if len(word)> 1 and word[-2] == 'es':
                #plural to singular
                word = word[:-2]
            if len(word)>0 and word[-1] == 's':
                #plural to singular
                word = word[:-1]
            return word
        exclude_list = ['hand','leg', 'left hand', 'right hand', 'man', 'woman', 'person', 'lady', 'they', 'ground', 'camera']
        max_n_words = 4
        noun_idxs = eval(sample['tag_noun'])[:max_n_words]
        noun_arr = torch.zeros([max_n_words]).fill_(0)
        words,noun_groups,counter = [],[],0

        for idx in noun_idxs: noun_groups += self.noun_pd.iloc[idx]['group']
        sentence_words =  caption.split()

        for word_i,word in enumerate(sentence_words[:-1]):
            word = p(word)
            two_word = ' '.join([p(sentence_words[word_i]),p(sentence_words[word_i+1])])

            if two_word in noun_groups and two_word not in exclude_list:
                words.append(two_word)
                noun_arr[counter] = self.all_nouns.index(two_word)

                counter +=1
            if counter >= max_n_words:
                break
        
        word_elements = []
        for w in words:
            word_elements+= w.split()

        for word_i,word in enumerate(sentence_words):
            if counter >= max_n_words:
                break
            word = p(word)
            if word in noun_groups and word not in word_elements and word not in exclude_list:
                words.append(word)
                noun_arr[counter] = self.all_nouns.index(word)
                counter +=1
        return words, noun_arr
    
    def _get_train_item(self, item):
        self.counter+=1
        
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption, noun_vec, verb_vec = self._get_caption(sample)
        rephrased_caption = self._get_rephrased_caption(sample, video_sec, caption)
        _, nouns = self.extract_noun(sample, caption)
        box, image_size = self.load_hand_object_box(sample)

        if self.video_params.get('disable', False):
            final = torch.zeros(1)
            valid = 0
        else:
            final,im_sz,crop_params,valid = self._get_video_frames(video_fp, video_sec, bound_sec, boxes=(box if self.crop_with_boxes else None))
        box = crop_boxes(box,crop_params,ori_im_sz=image_size,resize_target=224)

 
        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}

        # Scene-aware negative sampling
        if self.neg_param:
            # sample_neg = self.metadata[self.metadata.segment_id==sample.segment_id].sample(1).iloc[0]

            sample_negs = self.metadata_by_segment_id[sample.segment_id]
            sample_neg = sample_negs.sample(1).iloc[0]
            caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
            counter=0
            # resample if hard negative is the same as positive
            while sample_negs.shape[0]!=1 and  sample_neg['clip_start'] == sample['clip_start'] and counter <10:
                sample_neg = sample_negs.sample(1).iloc[0]
                caption_neg, noun_vec_neg, verb_vec_neg = self._get_caption(sample_neg)
                counter +=1

            video_fp_neg, video_sec_neg, bound_sec_neg = self._get_video_path(sample_neg)
            rephrased_caption_neg = self._get_rephrased_caption(sample_neg, video_sec_neg,caption_neg)
            box_neg, image_size_neg = self.load_hand_object_box(sample_neg)

            final_neg, im_sz_neg, crop_params_neg, valid  = self._get_video_frames(video_fp_neg, video_sec_neg, bound_sec_neg,  
                                                                               boxes=(box_neg if self.crop_with_boxes else None))
            box_neg = crop_boxes(box_neg,crop_params,ori_im_sz=image_size_neg,resize_target=224)
            _, nouns_neg = self.extract_noun(sample_neg, caption_neg)
        meta_arr_neg = {'raw_captions': caption_neg, 'paths': video_fp_neg, 'dataset': self.dataset_name}

        if self.neg_param:
            return {'video': final, 'text': caption,
                    'video_neg': final_neg, 'text_neg': caption_neg,
                    'meta': meta_arr, 'meta_neg': meta_arr_neg,
                    'noun_vec': noun_vec, 'verb_vec': verb_vec, 'nouns': nouns,
                    'noun_vec_neg': noun_vec_neg, 'verb_vec_neg': verb_vec_neg,'nouns_neg':nouns_neg,
                    'boxes':box, 'boxes_neg': box_neg,
                    'image_size': torch.tensor(im_sz), 'image_size_neg': torch.tensor(im_sz_neg),
                    'crop_params': crop_params, 'crop_params_neg': crop_params_neg,
                    'rephrased_text': rephrased_caption ,
                    'rephrased_text_neg': rephrased_caption_neg,
                    'data_item':item }

        else:
            return {'video': final, 'text': caption,
                'meta': meta_arr,
                'noun_vec': noun_vec, 'verb_vec': verb_vec,
                'clip_start': torch.tensor(sample['clip_start']),
                'clip_end': torch.tensor(sample['clip_end']),
                'boxes':box, 'image_size': torch.tensor(im_sz),
                'crop_params': crop_params,
                'valid': valid,
                'rephrased_text': rephrased_caption ,}


    def _get_val_item(self, item):
        item = item % len(self.metadata)
        itemMCQ = self.metadata[self.metakeys[item]]
        # itemMCQ = self.metadata[str(item)]

        answerIndex = itemMCQ['answer']
        sampleQuery = itemMCQ['query']
        textQuery, _, _ = self._get_caption(sampleQuery)

        sampleOptions = itemMCQ['choices']
        num_options = len(sampleOptions)
        textOptions = []
        videoOptions = torch.zeros([num_options, self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])

        videoIds, clipsStarts, clipEnds, imszs, boxes = [], [], [], [],[] 
        for id, option in enumerate(sampleOptions):
            sampleOptioni = sampleOptions[option]
            boxi, image_size = self.load_hand_object_box(sampleOptioni)
            boxes.append(boxi)

            video_fp, video_sec, bound_sec = self._get_video_path(sampleOptioni)
            caption, _, _ = self._get_caption(sampleOptioni)
            chunk_id = float(sampleOptions[str(id)]['narration_time']) // 600
            textOptions.append(caption)
            videoIds.append(sampleOptions[str(id)]['video_uid'] + f'/{int(chunk_id)}.mp4')
            clipsStarts.append(sampleOptions[str(id)]['clip_start'])
            clipEnds.append(sampleOptions[str(id)]['clip_end'])
            imgs,im_sz, _, _ = self._get_video_frames(video_fp, video_sec, bound_sec)            
            videoOptions[id] = imgs
            imszs.append(torch.tensor(im_sz))

        type =  itemMCQ['types']    # 1 for inter; 2 for intra

        data = {'video': videoOptions, 
                'text': textQuery, 
                'text_ops':textOptions, 
                'correct': answerIndex, 
                'type': type,
                'video_ids': videoIds,
                'clip_starts': clipsStarts,
                'clip_ends': clipEnds,
                'image_size':torch.stack(imszs),
                'boxes':torch.stack(boxes),
                }
        return data

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == 'train':
            return self._get_train_item(item)
        elif self.split in ['val', 'test']:
            return self._get_val_item(item)

def custom_collate(batch):
    elem = batch[0]
    combined_dict = {}
    for k  in [*elem.keys()]:
        if isinstance(elem[k], torch.Tensor):
            combined_dict[k] = torch.stack([b[k] for b in batch])
        elif isinstance(elem[k], list):
            combined_dict[k] = []
            for  b in batch:
                combined_dict[k].append(b[k])
        elif isinstance(elem[k], str):
            combined_dict[k] = []
            for b in batch:
                combined_dict[k].append(elem[k])
        elif isinstance(elem[k],int):
            combined_dict[k] = torch.tensor([elem[k] for b in batch])

    return combined_dict





if __name__ == "__main__":
    kwargs = dict(
        dataset_name="EgoClip_dataset",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="",
        meta_dir="",
        tsfms=init_video_transform_dict()['test'],
        reader='cv2_egoclip',
        split='train',
        neg_param=60
    )
    dataset = EgoClip_EgoMCQ(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())



