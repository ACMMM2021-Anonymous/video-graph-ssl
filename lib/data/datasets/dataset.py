import torch.utils.data as data

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


class BaseDataset(data.Dataset):
    def __init__(self, root_path, list_file, video_length=3,
                 new_length=1, sample_type='uniform', modality='RGB', pre_load='cv2',
                 image_tmpl='img_{:05d}.jpg', transform=None, use_adver=False, nsamples=40,
                 force_grayscale=False, random_shift=True, test_mode=False, num_clips=10):

        self.root_path = root_path
        self.list_file = list_file
        self.video_length = video_length
        self.sample_type = sample_type
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.use_adver = use_adver
        self.test_mode = test_mode
        self.num_clips = num_clips
        self.nsamples = nsamples
        self.pre_load = pre_load

        if self.modality == 'RGBDiff':
            self.new_length += 1

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
                y_img =  pil_loader(os.path.join(directory, self.image_tmpl.format('y', idx)))

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list  equal to video_length
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.video_length
        if average_duration > 0:
            offsets = np.multiply(list(range(self.video_length)), average_duration) + randint(average_duration, size=self.video_length)
        elif record.num_frames > self.video_length:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.video_length))
        else:
            offsets = np.zeros((self.video_length,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.video_length + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.video_length)
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

        offsets =np.array([int(v) for v in offsets])  

        return offsets + 1

    def _get_adver_indices(self, record, nsamples):
        """
        :return: list n sample, each equal to video_length
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.video_length
        if average_duration > 0:
            start_pos = np.multiply(list(range(self.video_length)), average_duration)
            ticks = [randint(average_duration, size=self.video_length) for i in range(nsamples)]
            offsets_set = [start_pos + i + 1 for i in ticks]
            offsets = []
            for off in offsets_set:
                offsets.extend(off)
        elif record.num_frames > self.video_length:
            ticks = [randint(record.num_frames - self.new_length + 1, size=self.video_length) for i in range(nsamples)]
            offsets = []
            for tick in ticks:
                offsets.extend(tick+1)
        else:
            offsets = []
            for i in range(nsamples):
                offsets.extend(np.zeros((self.video_length, ))+1)
        return offsets

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.video_length)

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
            if self.use_adver:
                segment_indices = self._get_adver_indices(record, self.nsamples)
            elif self.sample_type == 'uniform':
                segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
            elif self.sample_type == 'dense':
                segment_indices = self._get_dense_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_nclips_test_indices(record) if self.num_clips > 0 else  self._get_test_indices(record)

        return self.get_item(record, segment_indices)

    def get_item(self, record, indices):

        if self.root_path is not None:
            root_path = self.root_path
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(root_path+'/'+record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        process_data = self.transform(images)
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
