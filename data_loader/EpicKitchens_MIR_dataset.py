import os
import sys
import random
import pickle
import numpy as np
import pandas as pd

import torch
import decord

sys.path.insert(0, "../")

from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_video_transform_dict


def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(os.path.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids)
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        import ipdb; ipdb.set_trace()
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


class MultiInstanceRetrieval(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'retrieval_annotations/EPIC_100_retrieval_train.csv',
            'val': 'retrieval_annotations/EPIC_100_retrieval_test.csv',            # there is no test
            'test': 'retrieval_annotations/EPIC_100_retrieval_test.csv'
        }
        split_files_sentence = {
            'train': 'retrieval_annotations/EPIC_100_retrieval_train_sentence.csv',
            'val': 'retrieval_annotations/EPIC_100_retrieval_test_sentence.csv',  # there is no test
            'test': 'retrieval_annotations/EPIC_100_retrieval_test_sentence.csv'
        }
        target_split_fp = split_files[self.split]
        metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp))

        target_split_sentence_fp = split_files_sentence[self.split]
        metadata_sentence = pd.read_csv(os.path.join(self.meta_dir, target_split_sentence_fp))

        if self.split == 'train':
            path_relevancy = os.path.join(self.meta_dir, 'relevancy/caption_relevancy_EPIC_100_retrieval_train.pkl')
        elif self.split in ['val', 'test']:
            path_relevancy = os.path.join(self.meta_dir, 'relevancy/caption_relevancy_EPIC_100_retrieval_test.pkl')

        pkl_file = open(path_relevancy, 'rb')
        self.relevancy = 0.1
        self.relevancy_mat = pickle.load(pkl_file)
        self.metadata = metadata
        self.metadata_sentence = metadata_sentence


    def _get_video_path(self, sample, high_res =False):
        rel_video_fp = sample[2]
        pid = rel_video_fp.split('_')[0]
        full_video_fp = os.path.join(self.data_dir, pid, 'rgb_frames', rel_video_fp)

        return full_video_fp, rel_video_fp

    def _get_caption(self, idx, sample):
        # return sentence, relevancy score, idx
        if self.split == 'train':
            positive_list = np.where(self.relevancy_mat[idx] > self.relevancy)[0].tolist()
            if positive_list != []:
                pos = random.sample(positive_list, min(len(positive_list), 1))[0]
                if pos < len(self.metadata_sentence) and pos < self.relevancy_mat.shape[1]:
                    return self.metadata_sentence.iloc[pos][1], self.relevancy_mat[idx][pos], pos
            return sample[8], 1, 0

        elif self.split in ['val', 'test']:
            return sample[8], 1, -1

    def __getitem__(self, item):
        high_res = (self.video_params['num_frames'] == 16)
        high_res = False
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, rel_fp = self._get_video_path(sample, high_res=high_res)
        caption, relation, idx = self._get_caption(item, sample)

        fps_dict = torch.load(os.path.join(self.meta_dir,'fps_dict_256.pth'))
        video_fp = os.path.join(sample.participant_id, sample.video_id +'.MP4')
        start_timestamp, end_timestamp = datetime2sec(sample[4]), datetime2sec(sample[5])
        fps = fps_dict[os.path.join(self.data_dir,video_fp)]
        start_frame = int(np.round(fps * start_timestamp))
        end_frame = int(np.ceil(fps * end_timestamp))
        frame_ids = get_frame_ids(start_frame, end_frame, num_segments=self.video_params['num_frames'], jitter=False)
        imgs = video_loader_by_frames(self.data_dir, video_fp, frame_ids) # T, H, W, C
        imgs = imgs.permute(0,3,1,2) /255
   
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

        meta_arr = {'raw_captions': caption, 'paths': item, 'dataset': self.dataset_name}
        data = {'video': final, 'text': caption, 'meta': meta_arr, 'relation': relation, 'item_v': item, 'item_t': idx}
        return data



if __name__ == "__main__":
    tsfm_params = {
            "norm_mean": [108.3272985/255, 116.7460125/255, 104.09373615000001/255],
            "norm_std": [68.5005327/255, 66.6321579/255, 70.32316305/255]}
    kwargs = dict(
        dataset_name="MultiInstanceRetrieval",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 16,
        "loading": "lax"
        },
        data_dir="",
        meta_dir="",
        tsfms=init_video_transform_dict(
            norm_mean= [108.3272985/255, 116.7460125/255, 104.09373615000001/255],
            norm_std = [68.5005327/255, 66.6321579/255, 70.32316305/255])['test'],
        reader='cv2_epic',
        split='test'
    )
    dataset = MultiInstanceRetrieval(**kwargs)
    for i in range(100):
        item = dataset[i]
        print(item.keys())