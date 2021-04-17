import torch.utils.data as data
import torch

import os
import os.path
import numpy as np
from numpy.random import randint

from .utils import opencv_loader, pil_loader2, pil_loader

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class VisualDataset(data.Dataset):
    def __init__(self, 
                 root_path, 
                 list_file,
                 video_length=3, 
                 frame_interval=4, 
                 num_clips=1,
                 sample_type='uniform', 
                 modality='RGB', 
                 image_tmpl='img_{:05d}.jpg',
                 transform=None, 
                 mem_type='moco', 
                 temporal_jitter=False, 
                 pre_load='cv2',
                 force_grayscale=False, 
                 random_shift=True, 
                 test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.video_length = video_length
        self.sample_type = sample_type
        self.modality = modality
        self.frame_interval = frame_interval
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift  
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.mem_type = mem_type
        self.temporal_jitter = temporal_jitter
        self.pre_load = pre_load

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            if self.pre_load == 'cv2':
                return [opencv_loader(os.path.join(directory, self.image_tmpl.format(idx)))]
            else:
                return [pil_loader2(os.path.join(directory, self.image_tmpl.format(idx)))]
        elif self.modality == 'Flow':
            if self.pre_load == 'cv2':
                x_img = opencv_loader(os.path.join(directory, self.image_tmpl.format('x', idx)))
                y_img = opencv_loader(os.path.join(directory, self.image_tmpl.format('y', idx)))
            else:
                x_img = pil_loader(os.path.join(directory, self.image_tmpl.format('x', idx)))
                y_img = pil_loader(os.path.join(directory, self.image_tmpl.format('y', idx)))

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list  equal to video_length
        """

        average_duration = (record.num_frames + 1) // self.video_length
        if average_duration > 0:
            offsets = np.multiply(list(range(self.video_length)), average_duration) + randint(average_duration, size=self.video_length)
        elif record.num_frames > self.video_length:
            offsets = np.sort(randint(record.num_frames + 1, size=self.video_length))
        else:
            offsets = np.zeros((self.video_length,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.video_length - 1:
            tick = (record.num_frames + 1) / float(self.video_length)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.video_length)])
        else:
            offsets = np.zeros((self.video_length,))
        return offsets + 1

    def _get_dense_indices(self, record, step=4):
        expanded_sample_length = self.video_length * step  
        if record.num_frames >= expanded_sample_length:
            start_pos = randint(record.num_frames - expanded_sample_length + 1)
            offsets = range(start_pos, start_pos + expanded_sample_length, step)
        elif record.num_frames > self.video_length*(step//2):
            start_pos = randint(record.num_frames - self.video_length*(step//2) + 1)
            offsets = range(start_pos, start_pos + self.video_length*(step//2), (step//2))
        elif record.num_frames > self.video_length:
            start_pos = randint(record.num_frames - self.video_length + 1)
            offsets = range(start_pos, start_pos + self.video_length, 1)
        else:
            offsets = np.sort(randint(record.num_frames, size=self.video_length))

        offsets = np.array([int(v) for v in offsets])  

        # return offsets
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames + 1) / float(self.video_length)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.video_length)])

        return offsets + 1

    def _get_nclips_test_indices(self, record, step=4):
        tick = (record.num_frames - self.video_length*step + 1) / float(self.num_clips)
        sample_start_pos = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_clips)])
        offsets = []
        for p in sample_start_pos:
            offsets.extend(range(p, p+self.video_length*step, step))

        checked_offsets = []
        for f in offsets:
            new_f = int(f) + 1
            if new_f < 1:
                new_f = 1
            elif new_f >= record.num_frames:
                new_f = record.num_frames - 1
            checked_offsets.append(new_f)

        return checked_offsets

    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            if self.sample_type == 'uniform':
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                if self.temporal_jitter:
                    segment_indices = [self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
                                        for _ in range(2)]
            elif self.sample_type == 'dense':
                segment_indices = self._get_dense_indices(record, step=self.frame_interval)
                if self.temporal_jitter:
                    segment_indices = [self._get_dense_indices(record, step=self.frame_interval)
                                        for _ in range(2)]
        else:
            segment_indices = self._get_test_indices(record)

        process_data, label = self.get_item(record, segment_indices)
        return process_data, label, index

    def get_item(self, record, indices):

        if self.temporal_jitter:
            assert type(indices) is list, 'While using temporal jitter, indices should be list!'

        if self.root_path is not None:
            root_path = self.root_path
        
        if self.temporal_jitter:
            images_1, images_2 = list(), list() 
            for seg_ind in indices[0]:
                p = int(seg_ind)
                seg_imgs = self._load_image(root_path+'/'+record.path, p)
                images_1.extend(seg_imgs)
            for seg_ind in indices[1]:
                p = int(seg_ind)
                seg_imgs = self._load_image(root_path+'/'+record.path, p)
                images_2.extend(seg_imgs)
        else:
            images_1 = list()
            for seg_ind in indices:
                p = int(seg_ind)
                seg_imgs = self._load_image(root_path+'/'+record.path, p)
                images_1.extend(seg_imgs)

        process_data = self.transform(images_1)       
        if self.mem_type == 'moco' or self.mem_type == 'simsiam' or self.mem_type == 'bank':
            if self.temporal_jitter:
                process_data_1 = self.transform(images_2)
            else:
                process_data_1 = self.transform(images_1)
            process_data = torch.cat((process_data, process_data_1), dim=0)
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)

    def __repr__(self):
        fmt_str = '    Number of datapoints: {}\n'.format(self.__len__())
        if hasattr(self, 'root_path'):
            fmt_str += '    Root Location: {}\n'.format(self.root_path)
        tmp = '    Transforms (if any): '
        if hasattr(self, 'transform'):
            fmt_str += '{0}{1}\n'.format(
                tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
