# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen, swav=False, sfid=False):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    if opts.rgba:
        stats_real, stats_seg_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, swav=swav, sfid=sfid)

        (mu_real, sigma_real), (mu_seg_real, sigma_seg_real) = stats_real.get_mean_cov(), stats_seg_real.get_mean_cov()

        stats_gen, stats_seg_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

        (mu_gen, sigma_gen), (mu_seg_gen, sigma_seg_gen) = stats_gen.get_mean_cov(), stats_seg_gen.get_mean_cov()

        if opts.rank != 0:
            return float('nan'), float('nan')

        # images stats for fid
        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False)  # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))

        # mask stats for fid
        m_seg = np.square(mu_seg_gen - mu_seg_real).sum()
        s_seg, _ = scipy.linalg.sqrtm(np.dot(sigma_seg_gen, sigma_seg_real), disp=False)  # pylint: disable=no-member
        fid_seg = np.real(m_seg + np.trace(sigma_seg_gen + sigma_seg_real - s_seg * 2))
        return float(fid), float(fid_seg)
    else:
        mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, swav=swav, sfid=sfid).get_mean_cov()

        mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
            opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
            rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

        if opts.rank != 0:
            return float('nan')

        m = np.square(mu_gen - mu_real).sum()
        s, _ = scipy.linalg.sqrtm(np.dot(sigma_gen, sigma_real), disp=False) # pylint: disable=no-member
        fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
        return float(fid)

#----------------------------------------------------------------------------
