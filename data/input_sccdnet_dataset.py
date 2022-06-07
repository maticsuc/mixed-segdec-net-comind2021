import os
import numpy as np
from data.dataset import Dataset
from config import Config
from datetime import datetime

class SccdnetDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(SccdnetDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_samples(self, path_to_samples):
        samples = [i for i in sorted(os.listdir(os.path.join(path_to_samples, 'images')))]

        for sample in samples:
            id, file_type = sample.rsplit(".", 1)
            
            image_path = os.path.join(path_to_samples, 'images', sample)
            seg_mask_path = os.path.join(path_to_samples, 'masks', sample)
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, _ = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            seg_mask = self.to_tensor(seg_mask)

            if 'noncrack' in sample:
                self.neg_samples.append((image, seg_mask, True, image_path, seg_mask_path, id, False))
            else:
                self.pos_samples.append((image, seg_mask, True, image_path, seg_mask_path, id, True))
    
    def read_contents(self):

        self.pos_samples = list()
        self.neg_samples = list()

        if self.kind == 'TRAIN':
            self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train'))
        elif self.kind == 'TEST':
            self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'test'))

        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)

        self.len = self.num_pos + self.num_neg
        
        time = datetime.now().strftime("%d-%m-%y %H:%M")

        self.pos_weight = None

        if self.kind == 'TRAIN' and self.cfg.BCE_LOSS_W:
            neg = self.count_pixels(0)
            pos = self.count_pixels(1)
            self.pos_weight = neg / pos
            print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}, pos_weight: {self.pos_weight}")
        else:
            print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()