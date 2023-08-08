import numpy as np
import os.path as osp


import decord
import torch
import os

def datetime2sec(str):
    hh, mm, ss = str.split(':')
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def video_loader(root, vid, second, end_second=None, chunk_len=300, fps=30, clip_length=32, jitter=False):
    if chunk_len == -1:
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid)))
        second_offset = second
        if end_second is not None:
            end_second = min(end_second, len(vr) / vr.get_avg_fps())
        else:
            end_second = len(vr) / vr.get_avg_fps()
    else:
        chunk_start = int(second) // chunk_len * chunk_len
        second_offset = second - chunk_start
        vr = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start)))
    if fps == -1:
        fps = vr.get_avg_fps()

    # calculate frame_ids
    frame_offset = int(np.round(second_offset * fps))
    total_duration = max(int((end_second - second) * fps), clip_length)
    if chunk_len == -1:
        if end_second <= second:
            raise ValueError("end_second should be greater than second")
        else:
            frame_ids = get_frame_ids(frame_offset, min(frame_offset + total_duration, len(vr)), num_segments=clip_length, jitter=jitter)
    else:
        frame_ids = get_frame_ids(frame_offset, frame_offset + total_duration, num_segments=clip_length, jitter=jitter)

    # load frames
    if max(frame_ids) < len(vr):
        try:
            frames = vr.get_batch(frame_ids).asnumpy()
        except decord.DECORDError as error:
            print(error)
            frames = vr.get_batch([0] * len(frame_ids)).asnumpy()
    else:
        # find the remaining frames in the next chunk
        try:
            frame_ids_part1 = list(filter(lambda frame_id: frame_id < len(vr), frame_ids))
            frames_part1 = vr.get_batch(frame_ids_part1).asnumpy()
            vr2 = decord.VideoReader(osp.join(root, '{}.mp4'.format(vid), '{}.mp4'.format(chunk_start + chunk_len)))
            frame_ids_part2 = list(filter(lambda frame_id: frame_id >= len(vr), frame_ids))
            frame_ids_part2 = [min(frame_id % len(vr), len(vr2) - 1) for frame_id in frame_ids_part2]
            frames_part2 = vr2.get_batch(frame_ids_part2).asnumpy()
            frames = np.concatenate([frames_part1, frames_part2], axis=0)
        # the next chunk does not exist; the current chunk is the last one
        except (RuntimeError, decord.DECORDError) as error:
            print(error)
            frame_ids = get_frame_ids(min(frame_offset, len(vr) - 1), len(vr), num_segments=clip_length, jitter=jitter)
            frames = vr.get_batch(frame_ids).asnumpy()

    frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
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


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).to(torch.float32)
        frames = [frame for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)


class VideoCaptionDatasetBase(torch.utils.data.Dataset):
    def __init__(self, dataset, root, metadata, is_trimmed=True, anno_dir=''):
        self.dataset = dataset
        self.root = root
        self.is_trimmed = is_trimmed

        egtea_video_list = torch.load(os.path.join(anno_dir,'egtea_video_list.pth.tar'))
        len_dict = egtea_video_list['len_dict']

        vn_list, labels = [], []
        for row in open(osp.join(osp.dirname(metadata), 'action_idx.txt')):
            row = row.strip()
            vn = int(row.split(' ')[-1])
            vn_list.append(vn)
            narration = ' '.join(row.split(' ')[:-1])
            labels.append(narration.replace('_', ' ').lower())
        mapping_act2narration = {vn: narration for vn, narration in zip(vn_list, labels)}

        self.samples = []
        with open(metadata) as f:
            for row in f:
                clip_id, action_idx = row.strip().split(' ')[:2]
                video_id = '-'.join(clip_id.split('-')[:3])
                vid_relpath = osp.join(video_id, '{}.mp4'.format(clip_id))
                vid_fullpath = osp.join(self.root, video_id, '{}.mp4'.format(clip_id))
                self.samples.append((vid_relpath, 0, len_dict[vid_fullpath], mapping_act2narration[int(action_idx)]))
      

    def get_raw_item(self, i, is_training=True, num_clips=1, clip_length=32, clip_stride=2, sparse_sample=False,
                     narration_selection='random'):

        vid_path, start_frame, end_frame, sentence = self.samples[i]
        if is_training:
            assert num_clips == 1
            if end_frame < clip_length * clip_stride:
                frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                frames = torch.cat((frames, zeros), dim=0)
                frames = frames[::clip_stride]
            else:
                start_id = np.random.randint(0, end_frame - clip_length * clip_stride + 1)
                frame_ids = np.arange(start_id, start_id + clip_length * clip_stride, clip_stride)
                frames = video_loader_by_frames(self.root, vid_path, frame_ids)
        else:
            if end_frame < clip_length * clip_stride:
                frames = video_loader_by_frames(self.root, vid_path, list(np.arange(0, end_frame)))
                zeros = torch.zeros((clip_length * clip_stride - end_frame, *frames.shape[1:]))
                frames = torch.cat((frames, zeros), dim=0)
                frames = frames[::clip_stride]
                frames = frames.repeat(num_clips, 1, 1, 1)
            else:
                frame_ids = []
                for start_id in np.linspace(0, end_frame - clip_length * clip_stride, num_clips, dtype=int):
                    frame_ids.extend(np.arange(start_id, start_id + clip_length * clip_stride, clip_stride))
                frames = video_loader_by_frames(self.root, vid_path, frame_ids)
        return frames, sentence
       

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)


class VideoClassyDataset(VideoCaptionDatasetBase):
    def __init__(
        self, dataset, root, metadata, transform=None,
        is_training=True, label_mapping=None,
        num_clips=1,
        clip_length=32, clip_stride=2,
        sparse_sample=False,
        is_trimmed=True,
        anno_dir='',
    ):
        super().__init__(dataset, root, metadata, is_trimmed=is_trimmed, anno_dir=anno_dir)

        self.transform = transform
        self.is_training = is_training
        self.label_mapping = label_mapping
        self.num_clips = num_clips
        self.clip_length = clip_length
        self.clip_stride = clip_stride
        self.sparse_sample = sparse_sample
        self.anno_dir = anno_dir

    def __getitem__(self, i):
        frames, label = self.get_raw_item(
            i, is_training=self.is_training,
            num_clips=self.num_clips,
            clip_length=self.clip_length,
            clip_stride=self.clip_stride,
            sparse_sample=self.sparse_sample,
        )        
        # apply transformation
        if self.transform is not None:
            frames = self.transform(frames)

        if self.label_mapping is not None:
            if isinstance(label, list):
                # multi-label case
                res_array = np.zeros(len(self.label_mapping))
                for lbl in label:
                    res_array[self.label_mapping[lbl]] = 1.
                label = res_array
            else:
                label = self.label_mapping[label]

        return frames, label





def get_downstream_dataset(transform, tokenizer, args, subset='train', label_mapping=None):
    if subset == 'train':
        return VideoClassyDataset(
            args.dataset, args.root, args.metadata_train, transform,
            is_training=True, label_mapping=label_mapping,
            num_clips=args.num_clips,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
        )
    elif subset == 'val':
        return VideoClassyDataset(
            args.dataset, args.root, args.metadata_val, transform,
            is_training=False, label_mapping=label_mapping,
            num_clips=args.num_clips,
            clip_length=args.clip_length, clip_stride=args.clip_stride,
            sparse_sample=args.sparse_sample,
            is_trimmed=not args.dataset == 'charades_ego'
        )
    else:
        assert ValueError("subset should be either 'train' or 'val'")




def generate_label_map(action_idx_file):
    labels = []
    with open(action_idx_file) as f:
        for row in f:
            row = row.strip()
            narration = ' '.join(row.split(' ')[:-1])
            labels.append(narration.replace('_', ' ').lower())
            # labels.append(narration)
    mapping_vn2act = {label: i for i, label in enumerate(labels)}

    return labels, mapping_vn2act


if __name__ == "__main__":
    import torchvision.transforms as transforms
    from lavila_transforms import Permute, SpatialCrop, TemporalCrop
    import torchvision.transforms._transforms_video as transforms_video

    val_transform = transforms.Compose([
        Permute([3, 0, 1, 2]),  # T H W C -> C T H W
        transforms.Resize(224),
            transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305]),
        TemporalCrop(frames_per_clip=4, stride=4),
        SpatialCrop(crop_size=224, num_crops=1),
        ])

    val_dataset = VideoClassyDataset(
            'egtea', '/scratch/shared/beegfs/DATA/GTEA/cropped_clips', '/scratch/shared/beegfs/DATA/GTEA/test_split1.txt', val_transform,
            is_training=False, label_mapping=generate_label_map('./data/EGTEA/action_idx.txt')[1],
            num_clips=1,
            clip_length=4, clip_stride=16,
            sparse_sample=False,
            is_trimmed=True,
        )


    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=6, pin_memory=True, drop_last=False)

    import ipdb; ipdb.set_trace()