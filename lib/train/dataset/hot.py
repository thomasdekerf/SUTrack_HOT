import os
import glob
import numpy as np
import torch
from .base_video_dataset import BaseVideoDataset
from lib.train.data import jpeg4py_loader
from lib.train.admin import env_settings


class HOT(BaseVideoDataset):
    """HOT dataset loader.

    Expected directory structure:
        root/
            train/
                sequence_1/
                    *.jpg or *.png
                    groundtruth_rect.txt
                ...
            test/
                sequence_x/
                    *.jpg or *.png
                    groundtruth_rect.txt (optional for evaluation)
    The groundtruth file contains lines of 'cx cy w h'.
    """

    def __init__(self, root=None, split='train', image_loader=jpeg4py_loader,
                 multi_modal_vision=False, multi_modal_language=False, use_nlp=False):
        root = env_settings().hot_dir if root is None else root
        super().__init__('HOT', root, image_loader)
        self.split = split
        self.multi_modal_vision = multi_modal_vision
        self.multi_modal_language = multi_modal_language
        self.use_nlp = use_nlp
        self.base_path = os.path.join(self.root, split)
        if not os.path.isdir(self.base_path):
            raise ValueError(f'Invalid HOT {split} path: {self.base_path}')
        self.sequence_list = sorted([p for p in os.listdir(self.base_path)
                                     if os.path.isdir(os.path.join(self.base_path, p))])

    def get_name(self):
        return 'hot'

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.base_path, self.sequence_list[seq_id])

    def _get_frame_paths(self, seq_path):
        frames = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
        if len(frames) == 0:
            frames = sorted(glob.glob(os.path.join(seq_path, '*.png')))
        return frames

    def _read_bb_anno(self, seq_path):
        gt_path = os.path.join(seq_path, 'groundtruth_rect.txt')
        if not os.path.isfile(gt_path):
            return None
        gt = np.loadtxt(gt_path, delimiter=',')
        if gt.ndim == 1:
            gt = gt[None, :]
        # convert center format to top-left
        gt[:, 0] = gt[:, 0] - gt[:, 2] / 2
        gt[:, 1] = gt[:, 1] - gt[:, 3] / 2
        return torch.tensor(gt, dtype=torch.float32)

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)
        if bbox is None:
            bbox = torch.zeros(0, 4)
            valid = torch.zeros(0, dtype=torch.bool)
        else:
            valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)
        return {'bbox': bbox, 'valid': valid}

    def get_frames(self, seq_id, frame_ids, anno=None):
        seq_path = self._get_sequence_path(seq_id)
        frame_paths = self._get_frame_paths(seq_path)
        frame_list = [self.image_loader(frame_paths[i]) for i in frame_ids]

        if anno is None:
            anno = self.get_sequence_info(seq_id)

        anno_frames = {}
        for key, value in anno.items():
            if len(value) > 0:
                anno_frames[key] = [value[i].clone() for i in frame_ids]
            else:
                anno_frames[key] = [torch.zeros(4)] * len(frame_ids)
        obj_meta = {}
        return frame_list, anno_frames, obj_meta

    def get_num_sequences(self):
        return len(self.sequence_list)
