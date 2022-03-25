import os
import numpy as np
from data.dataset import Dataset
from config import Config
from datetime import datetime

class CrackSegmentationDataset(Dataset):
    def __init__(self, kind: str, cfg: Config):
        super(CrackSegmentationDataset, self).__init__(cfg.DATASET_PATH, cfg, kind)
        self.read_contents()

    def read_samples(self, path_to_samples, sample_kind):
        samples = [i for i in sorted(os.listdir(path_to_samples)) if 'GT' not in i]

        for sample in samples:
            part = sample.split(".")[0]
            
            image_path = path_to_samples + "/" + sample
            seg_mask_path = path_to_samples + "/" + part + "_GT." + sample.split(".")[1]
            
            image = self.read_img_resize(image_path, self.grayscale, self.image_size)
            image = self.to_tensor(image)

            seg_mask, positive = self.read_label_resize(seg_mask_path, self.image_size, self.cfg.DILATE)

            seg_mask = self.to_tensor(seg_mask)

            if sample_kind == 'pos':
                self.pos_samples.append((image, seg_mask, True, image_path, seg_mask_path, part, True))
            else:
                self.neg_samples.append((image, seg_mask, True, image_path, seg_mask_path, part, False))

    def read_contents(self):

        self.pos_samples = list()
        self.neg_samples = list()

        if self.cfg.DATASET == 'CRACK500':
            if self.kind == 'TEST':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'test'), 'pos')
            elif self.kind == 'TRAIN':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train'), 'pos')
                if self.cfg.USE_NEGATIVES:
                    self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train_negative'), 'neg')
            elif self.kind == 'VAL':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'val'), 'pos')
        else:
            if self.kind == 'TEST':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'test_positive'), 'pos')
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'test_negative'), 'neg')
            elif self.kind == 'TRAIN':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train_positive'), 'pos')
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'train_negative'), 'neg')
            elif self.kind == 'VAL':
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'val_positive'), 'pos')
                self.read_samples(os.path.join(self.cfg.DATASET_PATH, 'val_negative'), 'neg')

        self.num_pos = len(self.pos_samples)
        self.num_neg = len(self.neg_samples)

        self.len = self.num_pos + self.num_neg
        
        time = datetime.now().strftime("%d-%m-%y %H:%M")
        print(f"{time} {self.kind}: Number of positives: {self.num_pos}, Number of negatives: {self.num_neg}, Sum: {self.len}")

        self.init_extra()