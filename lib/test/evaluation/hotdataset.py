import os
import glob
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class HOTDataset(BaseDataset):
    """Evaluation dataset for HOT."""

    def __init__(self, split='test'):
        super().__init__()
        self.base_path = os.path.join(self.env_settings.hot_path, split)
        if not os.path.isdir(self.base_path):
            raise ValueError(f'Invalid HOT {split} path: {self.base_path}')
        self.sequence_list = self._get_sequences()

    def _get_sequences(self):
        seqs = []
        for seq_name in sorted(os.listdir(self.base_path)):
            seq_path = os.path.join(self.base_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            frame_paths = sorted(glob.glob(os.path.join(seq_path, '*.jpg')))
            if len(frame_paths) == 0:
                frame_paths = sorted(glob.glob(os.path.join(seq_path, '*.png')))
            gt_path = os.path.join(seq_path, 'groundtruth_rect.txt')
            if os.path.isfile(gt_path):
                gt = load_text(gt_path, delimiter=',', dtype=np.float64, backend='numpy')
                if gt.ndim == 1:
                    gt = gt[None, :]
                gt[:, 0] = gt[:, 0] - gt[:, 2] / 2
                gt[:, 1] = gt[:, 1] - gt[:, 3] / 2
            else:
                gt = np.zeros((1, 4))
            seqs.append(Sequence(seq_name, frame_paths, 'hot', gt))
        return seqs

    def get_sequence_list(self):
        return SequenceList(self.sequence_list)

    def __len__(self):
        return len(self.sequence_list)
