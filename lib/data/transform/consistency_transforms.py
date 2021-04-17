import cv2
import numpy as np
import random
import  numbers
import warnings
import math
import torch
import albumentations.augmentations.functional as F


class VideoToTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] (CxT, H, W)"""
    def __init__(self, backbone_type='3D'):
        if backbone_type == '3D' or backbone_type == '2D':
            self.backbone_type = backbone_type
        else:
            raise ValueError('Only 2D or 3D model is supported!')

    def __call__(self, clips):
        if isinstance(clips[0], np.ndarray):
            # handle numpy array
            if self.backbone_type == '3D':
                clips = np.stack(clips, axis=3)  # (H, W, C, T)
                clips = torch.from_numpy(clips.transpose(2, 3, 0, 1)).contiguous()
            if self.backbone_type == '2D':
                clips = np.concatenate(clips, axis=2)  # (H, W, TxC)
                clips = torch.from_numpy(clips.transpose(2, 0, 1)).contiguous()
        else:
            # handle PIL Image
            clips = [torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())) 
                    for img in clips]
            if self.backbone_type == '3D':
                clips = torch.stack(clips, dim=3)
                clips = clips.permute(2, 3, 0, 1).contiguous()
            if self.backbone_type == '2D':
                clips = torch.cat(clips, dim=2)
                clips = clips.permute(2, 0, 1).contiguous()

        if isinstance(clips, torch.ByteTensor):
            return clips.float()
        else:
            return clips

class VideoNormalize(object):
    def __init__(self, mean=(0.485, 0.456, 0.406), 
                std=(0.229, 0.224, 0.225),
                max_pixel_value=255.0):
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def normalize(self, img):
        mean = np.array(self.mean, dtype=np.float32)
        mean *= self.max_pixel_value

        std = np.array(self.std, dtype=np.float32)
        std *= self.max_pixel_value

        denominator = np.reciprocal(std, dtype=np.float32)

        img = img.astype(np.float32)
        img -= mean
        img *= denominator
        return img

    def __call__(self, clips):
        return [self.normalize(img) for img in clips]

class VideoRandomApply(object):
    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.p = p

    def __call__(self, clips):
        if random.random() < self.p:
            return self.transform(clips)
        else:
            return clips

class VideoRandomResizedCrop(object):    
    def __init__(self, size,
                scale=(0.08, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=cv2.INTER_LINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation

    def get_params(self, img):
        area = img.shape[0] * img.shape[1]

        for _attempt in range(10):
            target_area = random.uniform(*self.scale) * area
            log_ratio = (math.log(self.ratio[0]), math.log(self.ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))  # skipcq: PTC-W0028
            h = int(round(math.sqrt(target_area / aspect_ratio)))  # skipcq: PTC-W0028

            if 0 < w <= img.shape[1] and 0 < h <= img.shape[0]:
                i = random.randint(0, img.shape[0] - h)
                j = random.randint(0, img.shape[1] - w)
                return {
                    "crop_height": h,
                    "crop_width": w,
                    "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
                    "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
                }

        # Fallback to central crop
        in_ratio = img.shape[1] / img.shape[0]
        if in_ratio < min(self.ratio):
            w = img.shape[1]
            h = int(round(w / min(self.ratio)))
        elif in_ratio > max(self.ratio):
            h = img.shape[0]
            w = int(round(h * max(self.ratio)))
        else:  # whole image
            w = img.shape[1]
            h = img.shape[0]
        i = (img.shape[0] - h) // 2
        j = (img.shape[1] - w) // 2
        return {
            "crop_height": h,
            "crop_width": w,
            "h_start": i * 1.0 / (img.shape[0] - h + 1e-10),
            "w_start": j * 1.0 / (img.shape[1] - w + 1e-10),
        }

    def __call__(self, clips):
        params = self.get_params(clips[0])
        trans_clips = []
        for img in clips:
            img_crop = F.random_crop(img, params['crop_height'], params['crop_width'], 
                    params['h_start'], params['w_start'])
            img_resize = F.resize(img_crop, self.size[0], self.size[1], self.interpolation)
            trans_clips.append(img_resize)

        return trans_clips

class VideoRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        h_start, w_start = random.random(), random.random()
        return [F.random_crop(img, self.size[0], self.size[1], h_start, w_start)
                    for img in clips]

class VideoResize(object):
    def __init__(self, size, interpolation=cv2.INTER_LINEAR):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

        self.interpolation = interpolation

    def __call__(self, clips):
        return [F.resize(img, self.size[0], self.size[1], self.interpolation)
                for img in clips]

class VideoRandomRotate(object):
    """Rotate the input by an angle selected randomly from the uniform distribution. """
    def __init__(self, limit=90,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101
            ):
        self.limit = (-limit, limit)
        self.interpolation = interpolation
        self.border_mode = border_mode

    def __call__(self, clips):
        angle = random.uniform(self.limit[0], self.limit[1])
        return [F.rotate(img, angle, self.interpolation, self.border_mode)
                for img in clips]

class VideoRandomRotate90(object):
    """Randomly rotate the input by 90 degree zeros or more times"""
    def __call__(self, clips):
        factor = random.randint(0, 3)
        return [np.ascontiguousarray(np.rot90(img, factor)) for img in clips]


class VideoGaussianNoise(object):
    def __init__(self, var_limit=(10.0, 50.0), mean=0):
        if isinstance(var_limit, (tuple, list)):
            if var_limit[0] < 0:
                raise ValueError("Lower var_limit should be non negative.")
            if var_limit[1] < 0:
                raise ValueError("Upper var_limit should be non negative.")
            self.var_limit = var_limit
        elif isinstance(var_limit, (int, float)):
            if var_limit < 0:
                raise ValueError("var_limit should be non negative.")

            self.var_limit = (0, var_limit)
        else:
            raise TypeError(
                "Expected var_limit type to be one of (int, float, tuple, list), got {}".format(type(var_limit))
            )

        self.mean = mean

    def get_params(self, img):
        var = random.uniform(self.var_limit[0], self.var_limit[1])
        sigma = var ** 0.5
        random_state = np.random.RandomState(random.randint(0, 2 ** 32 - 1))

        gauss = random_state.normal(self.mean, sigma, img.shape)
        return {"gauss": gauss}

    def __call__(self, clips):
        params = self.get_params(clips[0])
        return [F.gauss_noise(img, gauss=params['gauss']) for img in clips]

class VideoGaussianBlur(object):
    """Blur the input image using using a Gaussian filter with a random kernel size."""
    def __init__(self, blur_limit=(3, 7), sigma_limit=0):
        if isinstance(blur_limit, numbers.Number):
            self.blur_limit = (0, blur_limit)
        else:
            self.blur_limit = blur_limit

        if isinstance(sigma_limit, numbers.Number):
            self.sigma_limit = (0, sigma_limit)
        else:
            self.sigma_limit = sigma_limit
            
        if self.blur_limit[0] == 0 and self.sigma_limit[0] == 0:
            self.blur_limit = 3, max(3, self.blur_limit[1])
            warnings.warn(
                "blur_limit and sigma_limit minimum value can not be both equal to 0. "
                "blur_limit minimum value changed to 3."
            )

        if (self.blur_limit[0] != 0 and self.blur_limit[0] % 2 != 1) or (
            self.blur_limit[1] != 0 and self.blur_limit[1] % 2 != 1
        ):
            raise ValueError("GaussianBlur supports only odd blur limits.")

    def get_params(self):
        ksize = np.random.randint(self.blur_limit[0], self.blur_limit[1] + 1)
        if ksize != 0 and ksize % 2 != 1:
            ksize = (ksize + 1) % (self.blur_limit[1] + 1)

        return {"ksize": ksize, "sigma": random.uniform(*self.sigma_limit)}

    def __call__(self, clips):
        params = self.get_params()
        return [F.gaussian_blur(img, ksize=params['ksize'], sigma=params['sigma']) 
                    for img in clips]

class VideoRandomGrayScale(object):
    """Convert the input RGB image to grayscale. If the mean pixel value for the resulting image is greater
    than 127, invert the resulting grayscale image."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clips):
        if random.random() < self.p:
            return [F.to_gray(img) for img in clips]
        else:
            return clips

class VideoRandomColorJitter(object):
    """Randomly changes the brightness, contrast, and saturation of an image. Compared to ColorJitter from torchvision,
    this transform gives a little bit different results because Pillow (used in torchvision) and OpenCV (used in
    Albumentations) transform an image to HSV format by different formulas. Another difference - Pillow uses uint8
    overflow, but we use value saturation.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0 <= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
    """
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
        self.brightness = self.__check_values(brightness, "brightness")
        self.contrast = self.__check_values(contrast, "contrast")
        self.saturation = self.__check_values(saturation, "saturation")
        self.hue = self.__check_values(hue, "hue", offset=0, bounds=[-0.5, 0.5], clip=False)

    @staticmethod
    def __check_values(value, name, offset=1, bounds=(0, float("inf")), clip=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [offset - value, offset + value]
            if clip:
                value[0] = max(value[0], 0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bounds[0] <= value[0] <= value[1] <= bounds[1]:
                raise ValueError("{} values should be between {}".format(name, bounds))
        else:
            raise TypeError("{} should be a single number or a list/tuple with length 2.".format(name))

        return value

    def get_params(self):
        brightness = random.uniform(self.brightness[0], self.brightness[1])
        contrast = random.uniform(self.contrast[0], self.contrast[1])
        saturation = random.uniform(self.saturation[0], self.saturation[1])
        hue = random.uniform(self.hue[0], self.hue[1])

        transforms = [
            lambda x: F.adjust_brightness_torchvision(x, brightness),
            lambda x: F.adjust_contrast_torchvision(x, contrast),
            lambda x: F.adjust_saturation_torchvision(x, saturation),
            lambda x: F.adjust_hue_torchvision(x, hue),
        ]
        random.shuffle(transforms)

        return {"transforms": transforms}

    def __call__(self, clips):
        transforms = self.get_params()['transforms']
        trans_clips = []
        for img in clips:
            for transform in transforms:
                img = transform(img)
            trans_clips.append(img)
        return trans_clips

class VideoCenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, clips):
        return [F.center_crop(img, self.size[0], self.size[1]) for img in clips]

class VideoRandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clips):
        if random.random() < self.p:
            if clips[0].ndim == 3 and clips[0].shape[2] > 1 and clips[0].dtype == np.uint8:
            # Opencv is faster than numpy only in case of
            # non-gray scale 8bits images
                return [F.hflip_cv2(img) for img in clips]

            return [F.hflip(img) for img in clips]
        else:
            return clips

class VideoMultiScaleCrop(object):
    """ A crop of size is selected from scales of the original size.
    A position of cropping is randomly selected from 4 corners and 1 center,
    or a position of cropping is randomly selected from 13 position while more_fix_crop
    This crop is finally resized to given size.
    1. Randomly select 1 position from 4 corners or center
    2. Randomly select 1 crop scale from 5 options ranging from 0.5 to 1
    Args:
        input_size: int, tuple
            new size of input image
        scales: list, default = [1, 0.875, 0.75, 0.66]
            list of scales to crop at
        max_distort: int default = 1
            Distortion limit
        fix_crop: bool, default = True
            Crop at fixed offset
        more_fix_crop: bool default = True
            More fix crop
    """
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, .875, .75, .66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size, input_size]
        self.interpolation = cv2.INTER_LINEAR

    def __call__(self, clips):
        im_size = clips[0].shape[0:2]
        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)

        trans_clips = []
        for img in clips:
            img = img[offset_h : offset_h + crop_h, offset_w : offset_w + crop_w]
            resize_img = F.resize(img, self.input_size[0], self.input_size[1], self.interpolation)
            trans_clips.append(resize_img)

        return trans_clips

    def _sample_crop_size(self, im_size):
        img_h, img_w = im_size[0], im_size[1]

        # find a crop size
        base_size = min(img_w, img_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [
            self.input_size[1] if abs(x - self.input_size[1]) < 3 else x
            for x in crop_sizes
        ]
        crop_w = [
            self.input_size[0] if abs(x - self.input_size[0]) < 3 else x
            for x in crop_sizes
        ]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        rand_idx = np.random.randint(len(pairs))
        crop_pair = pairs[rand_idx]
        if not self.fix_crop:
            w_offset = np.random.randint(0, img_w - crop_pair[0])
            h_offset = np.random.randint(0, img_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                img_w, img_h, crop_pair[0], crop_pair[1]
            )

        return crop_pair[0], crop_pair[1], int(w_offset), int(h_offset)

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(
            self.more_fix_crop, image_w, image_h, crop_w, crop_h
        )
        rand_idx = np.random.randint(len(offsets))
        return offsets[rand_idx]

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) / 4
        h_step = (image_h - crop_h) / 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter

        return ret

class VideoOverSampleCrop(object):
    # 5 or 10 crop
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = VideoResize(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, clips):

        if self.scale_worker is not None:
            clips = self.scale_worker(clips)

        image_h, image_w = clips[0].shape[0:2]
        crop_h, crop_w = self.crop_size

        offsets = VideoMultiScaleCrop.fill_fix_offset(False, image_w, image_h, crop_w, crop_h)
        oversample_clips = list()
        for offset_w, offset_h in offsets:
            normal_clips = list()
            flip_clips = list()
            for img in clips:
                crop = img[int(offset_h) : int(offset_h) + crop_h, int(offset_w) : int(offset_w) + crop_w]
                normal_clips.append(crop)
                if clips[0].ndim == 3 and clips[0].shape[2] > 1 and clips[0].dtype == np.uint8:
                    # Opencv is faster than numpy only in case of
                    # non-gray scale 8bits images
                    flip_clips.append(F.hflip_cv2(crop))
                else:
                    flip_clips.append(F.hflip(crop))

            oversample_clips.extend(normal_clips)
            if self.flip:
                oversample_clips.extend(flip_clips)
        return oversample_clips

class VideoFullResSample(object):
    # 3 or 6 crop
    def __init__(self, crop_size, scale_size=None, flip=True):
        self.crop_size = crop_size if not isinstance(crop_size, int) else (crop_size, crop_size)

        if scale_size is not None:
            self.scale_worker = VideoResize(scale_size)
        else:
            self.scale_worker = None
        self.flip = flip

    def __call__(self, clips):

        if self.scale_worker is not None:
            clips = self.scale_worker(clips)

        image_h, image_w = clips[0].shape[0:2]
        crop_h, crop_w = self.crop_size

        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        offsets = list()
        offsets.append((0 * w_step, 2 * h_step))  # left
        offsets.append((4 * w_step, 2 * h_step))  # right
        offsets.append((2 * w_step, 2 * h_step))  # center

        oversample_clips = list()
        for offset_w, offset_h in offsets:
            normal_clips = list()
            flip_clips = list()
            for img in clips:
                crop = img[int(offset_h) : int(offset_h) + crop_h, int(offset_w) : int(offset_w) + crop_w]
                normal_clips.append(crop)
                if self.flip:
                    if clips[0].ndim == 3 and clips[0].shape[2] > 1 and clips[0].dtype == np.uint8:
                        flip_clips.append(F.hflip_cv2(crop))
                    else:
                        flip_clips.append(F.hflip(crop))

            oversample_clips.extend(normal_clips)
            oversample_clips.extend(flip_clips)
        return oversample_clips

class VideoTemporalShuffle(object):
    
    def __call__(self, clips):
        index_list = list(range(len(clips)))
        random.shuffle(index_list)
        return clips[index_list]
