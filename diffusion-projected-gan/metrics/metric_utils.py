# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Miscellaneous utilities used internally by the quality metrics."""

import os
import time
import hashlib
import pickle
import copy
import uuid
import numpy as np
import torch
import torchvision.transforms as transforms
import dnnlib
from tqdm import tqdm

#----------------------------------------------------------------------------

class MetricOptions:
    def __init__(self, G=None, G_kwargs={}, dataset_kwargs={}, num_gpus=1, rank=0, device=None, progress=None, cache=True, run_dir=None, cur_nimg=None, snapshot_pkl=None):
        assert 0 <= rank < num_gpus
        self.G              = G
        self.G_kwargs       = dnnlib.EasyDict(G_kwargs)
        self.dataset_kwargs = dnnlib.EasyDict(dataset_kwargs)
        self.num_gpus       = num_gpus
        self.rank           = rank
        self.device         = device if device is not None else torch.device('cuda', rank)
        self.progress       = progress.sub() if progress is not None and rank == 0 else ProgressMonitor()
        self.cache          = cache
        self.run_dir = run_dir
        self.cur_nimg = cur_nimg
        self.snapshot_pkl = snapshot_pkl
        self.rgba = dataset_kwargs['rgba']
        self.rgba_mode = dataset_kwargs['rgba_mode']
        self.imnet_norm = dataset_kwargs['imnet_norm']

#----------------------------------------------------------------------------

_feature_detector_cache = dict()

def get_feature_detector_name(url):
    return os.path.splitext(url.split('/')[-1])[0]

def get_feature_detector(url, device=torch.device('cpu'), num_gpus=1, rank=0, verbose=False):
    assert 0 <= rank < num_gpus
    key = (url, device)
    if key not in _feature_detector_cache:
        is_leader = (rank == 0)
        if not is_leader and num_gpus > 1:
            torch.distributed.barrier() # leader goes first
        with dnnlib.util.open_url(url, verbose=(verbose and is_leader)) as f:
            _feature_detector_cache[key] = pickle.load(f).to(device)
        if is_leader and num_gpus > 1:
            torch.distributed.barrier() # others follow
    return _feature_detector_cache[key]

#----------------------------------------------------------------------------

def iterate_random_labels(opts, batch_size):
    if opts.G.c_dim == 0:
        c = torch.zeros([batch_size, opts.G.c_dim], device=opts.device)
        while True:
            yield c
    else:
        dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
        while True:
            c = [dataset.get_label(np.random.randint(len(dataset))) for _i in range(batch_size)]
            c = torch.from_numpy(np.stack(c)).pin_memory().to(opts.device)
            yield c

#----------------------------------------------------------------------------

class FeatureStats:
    def __init__(self, capture_all=False, capture_mean_cov=False, max_items=None):
        self.capture_all = capture_all
        self.capture_mean_cov = capture_mean_cov
        self.max_items = max_items
        self.num_items = 0
        self.num_features = None
        self.all_features = None
        self.raw_mean = None
        self.raw_cov = None

    def set_num_features(self, num_features):
        if self.num_features is not None:
            assert num_features == self.num_features
        else:
            self.num_features = num_features
            self.all_features = []
            self.raw_mean = np.zeros([num_features], dtype=np.float64)
            self.raw_cov = np.zeros([num_features, num_features], dtype=np.float64)

    def is_full(self):
        return (self.max_items is not None) and (self.num_items >= self.max_items)

    def append(self, x):
        x = np.asarray(x, dtype=np.float32)
        assert x.ndim == 2
        if (self.max_items is not None) and (self.num_items + x.shape[0] > self.max_items):
            if self.num_items >= self.max_items:
                return
            x = x[:self.max_items - self.num_items]

        self.set_num_features(x.shape[1])
        self.num_items += x.shape[0]
        if self.capture_all:
            self.all_features.append(x)
        if self.capture_mean_cov:
            x64 = x.astype(np.float64)
            self.raw_mean += x64.sum(axis=0)
            self.raw_cov += x64.T @ x64

    def append_torch(self, x, num_gpus=1, rank=0):
        assert isinstance(x, torch.Tensor) and x.ndim == 2
        assert 0 <= rank < num_gpus
        if num_gpus > 1:
            ys = []
            for src in range(num_gpus):
                y = x.clone()
                torch.distributed.broadcast(y, src=src)
                ys.append(y)
            x = torch.stack(ys, dim=1).flatten(0, 1) # interleave samples
        self.append(x.cpu().numpy())

    def get_all(self):
        assert self.capture_all
        return np.concatenate(self.all_features, axis=0)

    def get_all_torch(self):
        return torch.from_numpy(self.get_all())

    def get_mean_cov(self):
        assert self.capture_mean_cov
        mean = self.raw_mean / self.num_items
        cov = self.raw_cov / self.num_items
        cov = cov - np.outer(mean, mean)
        return mean, cov

    def save(self, pkl_file):
        with open(pkl_file, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @staticmethod
    def load(pkl_file):
        with open(pkl_file, 'rb') as f:
            s = dnnlib.EasyDict(pickle.load(f))
        obj = FeatureStats(capture_all=s.capture_all, max_items=s.max_items)
        obj.__dict__.update(s)
        return obj

#----------------------------------------------------------------------------

class ProgressMonitor:
    def __init__(self, tag=None, num_items=None, flush_interval=1000, verbose=False, progress_fn=None, pfn_lo=0, pfn_hi=1000, pfn_total=1000):
        self.tag = tag
        self.num_items = num_items
        self.verbose = verbose
        self.flush_interval = flush_interval
        self.progress_fn = progress_fn
        self.pfn_lo = pfn_lo
        self.pfn_hi = pfn_hi
        self.pfn_total = pfn_total
        self.start_time = time.time()
        self.batch_time = self.start_time
        self.batch_items = 0
        if self.progress_fn is not None:
            self.progress_fn(self.pfn_lo, self.pfn_total)

    def update(self, cur_items):
        assert (self.num_items is None) or (cur_items <= self.num_items)
        if (cur_items < self.batch_items + self.flush_interval) and (self.num_items is None or cur_items < self.num_items):
            return
        cur_time = time.time()
        total_time = cur_time - self.start_time
        time_per_item = (cur_time - self.batch_time) / max(cur_items - self.batch_items, 1)
        if (self.verbose) and (self.tag is not None):
            print(f'{self.tag:<19s} items {cur_items:<7d} time {dnnlib.util.format_time(total_time):<12s} ms/item {time_per_item*1e3:.2f}')
        self.batch_time = cur_time
        self.batch_items = cur_items

        if (self.progress_fn is not None) and (self.num_items is not None):
            self.progress_fn(self.pfn_lo + (self.pfn_hi - self.pfn_lo) * (cur_items / self.num_items), self.pfn_total)

    def sub(self, tag=None, num_items=None, flush_interval=1000, rel_lo=0, rel_hi=1):
        return ProgressMonitor(
            tag             = tag,
            num_items       = num_items,
            flush_interval  = flush_interval,
            verbose         = self.verbose,
            progress_fn     = self.progress_fn,
            pfn_lo          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_lo,
            pfn_hi          = self.pfn_lo + (self.pfn_hi - self.pfn_lo) * rel_hi,
            pfn_total       = self.pfn_total,
        )

#----------------------------------------------------------------------------

def compute_feature_stats_for_dataset(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, data_loader_kwargs=None, max_items=None, swav=False, sfid=False,  **stats_kwargs):
    dataset = dnnlib.util.construct_class_by_name(**opts.dataset_kwargs)
    if data_loader_kwargs is None:
        data_loader_kwargs = dict(pin_memory=True, num_workers=3, prefetch_factor=2)

    if opts.imnet_norm:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    # Try to lookup from cache.
    cache_file = None
    cache_seg_file = None
    if opts.cache:
        det_name = get_feature_detector_name(detector_url)

        # Choose cache file name.
        args = dict(dataset_kwargs=opts.dataset_kwargs, detector_url=detector_url, detector_kwargs=detector_kwargs, stats_kwargs=stats_kwargs)
        md5 = hashlib.md5(repr(sorted(args.items())).encode('utf-8'))
        if opts.rgba:
            cache_seg_tag = f'{dataset.name}-SEG-{det_name}-{md5.hexdigest()}'
            cache_seg_file = os.path.join('.', 'dnnlib', 'gan-metrics', cache_seg_tag + '.pkl')
        cache_tag = f'{dataset.name}-{det_name}-{md5.hexdigest()}'
        cache_file = os.path.join('.', 'dnnlib', 'gan-metrics', cache_tag + '.pkl')
        # cache_file = dnnlib.make_cache_dir_path('gan-metrics', cache_tag + '.pkl')

        # Check if the file exists (all processes must agree).
        if opts.rgba:
            flag = os.path.isfile(cache_file) if opts.rank == 0 else False
            flag_seg = os.path.isfile(cache_seg_file) if opts.rank == 0 else False
            if opts.num_gpus > 1:
                flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
                flag_seg = torch.as_tensor(flag_seg, dtype=torch.float32, device=opts.device)
                torch.distributed.broadcast(tensor=flag, src=0)
                torch.distributed.broadcast(tensor=flag_seg, src=0)
                flag = (float(flag.cpu()) != 0)
                flag_seg = (float(flag_seg.cpu()) != 0)

            # Load only if we have both.
            if flag and flag_seg:
                return FeatureStats.load(cache_file), FeatureStats.load(cache_seg_file)

        else:
            flag = os.path.isfile(cache_file) if opts.rank == 0 else False
            if opts.num_gpus > 1:
                flag = torch.as_tensor(flag, dtype=torch.float32, device=opts.device)
                torch.distributed.broadcast(tensor=flag, src=0)
                flag = (float(flag.cpu()) != 0)

            # Load.
            if flag:
                return FeatureStats.load(cache_file)

    print('Calculating the stats for this dataset the first time\n')
    if opts.rgba:
        print(f'Saving image stats to {cache_file}')
        print(f'Saving seg stats to {cache_seg_file}')
    else:
        print(f'Saving them to {cache_file}')

    # Initialize.
    num_items = len(dataset)
    if max_items is not None:
        num_items = min(num_items, max_items)
    if opts.rgba:
        stats = FeatureStats(max_items=num_items, **stats_kwargs)
        stats_seg = FeatureStats(max_items=num_items, **stats_kwargs)
    else:
        stats = FeatureStats(max_items=num_items, **stats_kwargs)

    progress = opts.progress.sub(tag='dataset features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)
    if opts.rgba:
        progress_seg = opts.progress.sub(tag='dataset seg features', num_items=num_items, rel_lo=rel_lo, rel_hi=rel_hi)

    # get detector
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    item_subset = [(i * opts.num_gpus + opts.rank) % num_items for i in range((num_items - 1) // opts.num_gpus + 1)]
    lo, hi = 0, 255
    if opts.rgba:
        for images, _labels in tqdm(
                torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size,
                                            **data_loader_kwargs)):
            # separate the img and mask, make mask rgb
            raw_images = torch.empty_like(images[:, :-1, :, :])
            raw_masks = torch.empty_like(images[:, :-1, :, :])

            # cast the values
            raw_images[:] = images[:, :-1, :, :][:]
            raw_masks[:] = images[:, -1:, :, :].repeat([1, 3, 1, 1])[:]

            if opts.imnet_norm: # try this
                raw_images = normalize(raw_images/255.)
                raw_masks = normalize(raw_masks/255.)

                # # scale the images to 0-255
                # images = (raw_images - lo) * (255 / (hi - lo))
                # masks = (raw_masks - lo) * (255 / (hi - lo))
                #
                # # round and clamp as needed
                # raw_images = raw_images.round().clamp(0, 255).to(torch.uint8)
                # raw_masks = raw_masks.round().clamp(0, 255).to(torch.uint8)

            else:
                # round and clamp as needed
                raw_images.to(torch.uint8)
                raw_masks.to(torch.uint8)

            with torch.no_grad():
                img_features = detector(raw_images.to(opts.device), **detector_kwargs)
                mask_features = detector(raw_masks.to(opts.device), **detector_kwargs)

            stats.append_torch(img_features, num_gpus=opts.num_gpus, rank=opts.rank)
            stats_seg.append_torch(mask_features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.num_items)
            progress_seg.update(stats_seg.num_items)

        # Save to cache.
        if cache_file is not None and opts.rank == 0:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = cache_file + '.' + uuid.uuid4().hex
            stats.save(temp_file)
            os.replace(temp_file, cache_file)  # atomic

        if cache_seg_file is not None and opts.rank == 0:
            os.makedirs(os.path.dirname(cache_seg_file), exist_ok=True)
            temp_seg_file = cache_seg_file + '.' + uuid.uuid4().hex
            stats_seg.save(temp_seg_file)
            os.replace(temp_seg_file, cache_seg_file)  # atomic

        return stats, stats_seg
    else:
        for images, _labels in tqdm(torch.utils.data.DataLoader(dataset=dataset, sampler=item_subset, batch_size=batch_size, **data_loader_kwargs)):
            if images.shape[1] == 1: # assumes single channel to rgb, adjust this all later
                images = images.repeat([1, 3, 1, 1])

            if opts.imnet_norm: # try this
                images = normalize(images/255.)
            else:
                # round and clamp as needed
                images = images.round().clamp(0, 255).to(torch.uint8)

            with torch.no_grad():
                features = detector(images.to(opts.device), **detector_kwargs)

            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.num_items)

        # Save to cache.
        if cache_file is not None and opts.rank == 0:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            temp_file = cache_file + '.' + uuid.uuid4().hex
            stats.save(temp_file)
            os.replace(temp_file, cache_file) # atomic
        return stats

#----------------------------------------------------------------------------

def compute_feature_stats_for_generator(opts, detector_url, detector_kwargs, rel_lo=0, rel_hi=1, batch_size=64, batch_gen=None, swav=False, sfid=False, **stats_kwargs):
    if batch_gen is None:
        batch_gen = min(batch_size, 4)
    assert batch_size % batch_gen == 0

    # Setup generator and labels.
    G = copy.deepcopy(opts.G).eval().requires_grad_(False).to(opts.device)
    c_iter = iterate_random_labels(opts=opts, batch_size=batch_gen)

    if opts.imnet_norm:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])

    # Initialize.
    # we want to generate stats for both images and masks if we are in seg mode
    if opts.rgba:
        # image stats
        stats = FeatureStats(**stats_kwargs)
        # mask stats
        stats_seg = FeatureStats(**stats_kwargs)
        assert stats.max_items is not None
        assert stats_seg.max_items is not None
    else:
        stats = FeatureStats(**stats_kwargs)
        assert stats.max_items is not None

    progress = opts.progress.sub(tag='generator features', num_items=stats.max_items, rel_lo=rel_lo, rel_hi=rel_hi)
    if opts.rgba:
        progress_seg = opts.progress.sub(tag='generator seg features', num_items=stats_seg.max_items,
                                         rel_lo=rel_lo, rel_hi=rel_hi)

    # get detector
    detector = get_feature_detector(url=detector_url, device=opts.device, num_gpus=opts.num_gpus, rank=opts.rank, verbose=progress.verbose)

    # Main loop.
    # need to specify for rgba so that we can get stats for masks
    lo, hi = -1, 1
    if opts.rgba:
        while not stats.is_full():
            images_full = []
            masks_full = []
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                # img = G(z=z, c=next(c_iter), truncation_psi=0.1, **opts.G_kwargs)

                # get the image outputs
                images = G(z=z, c=next(c_iter), **opts.G_kwargs)

                # separate the img and mask, make mask rgb
                raw_images = torch.empty_like(images[:, :-1, :, :])
                raw_masks = torch.empty_like(images[:, :-1, :, :])

                # cast the values
                raw_images[:] = images[:, :-1, :, :][:]
                raw_masks[:] = images[:, -1:, :, :].repeat([1, 3, 1, 1])[:]

                # scale the images to 0-255
                images_out = (raw_images - lo) * (255 / (hi - lo))
                masks_out = (raw_masks - lo) * (255 / (hi - lo))

                # round and clamp as needed
                images_out = images_out.round().clamp(0, 255).to(torch.uint8)
                masks_out = masks_out.round().clamp(0, 255).to(torch.uint8)

                # normalize
                if opts.imnet_norm:  # try this
                    images_out = normalize(images_out/255.)
                    masks_out = normalize(masks_out/255.)

                    # append them as needed
                    images_full.append(images_out)
                    masks_full.append(masks_out)

                else:
                    # append them as needed
                    images_full.append(images_out)
                    masks_full.append(masks_out)

            # get the input tensors
            images = torch.cat(images_full)
            masks = torch.cat(masks_full)

            # get the features
            with torch.no_grad():
                img_features = detector(images.to(opts.device), **detector_kwargs)
                mask_features = detector(masks.to(opts.device), **detector_kwargs)

            # append the features to stats
            stats.append_torch(img_features, num_gpus=opts.num_gpus, rank=opts.rank)
            stats_seg.append_torch(mask_features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.num_items)
            progress_seg.update(stats_seg.num_items)

        return stats, stats_seg

    else:
        while not stats.is_full():
            images = []
            for _i in range(batch_size // batch_gen):
                z = torch.randn([batch_gen, G.z_dim], device=opts.device)
                # img = G(z=z, c=next(c_iter), truncation_psi=0.1, **opts.G_kwargs)

                # get the image outputs
                img = G(z=z, c=next(c_iter), **opts.G_kwargs)

                # this shouldnt happen with generator? but will leave it...
                if img.shape[1] == 1:
                    img = img.repeat([1, 3, 1, 1])

                # scale the images to 0-255
                img = (img - lo) * (255 / (hi - lo))

                # round and clamp as needed
                img = img.round().clamp(0, 255).to(torch.uint8)

                if opts.imnet_norm:  # try this
                    img = normalize(img/255.)
                    images.append(img)

                else:
                    images.append(img)

            # get the input tensor
            images = torch.cat(images)

            with torch.no_grad():
                features = detector(images.to(opts.device), **detector_kwargs)

            stats.append_torch(features, num_gpus=opts.num_gpus, rank=opts.rank)
            progress.update(stats.num_items)

        return stats
