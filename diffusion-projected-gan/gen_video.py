# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate lerp videos using pretrained network pickle."""

import copy
import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import imageio
import numpy as np
import scipy.interpolate
import torch
from tqdm import tqdm

import cv2

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import legacy

#----------------------------------------------------------------------------

def layout_grid(img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img

#----------------------------------------------------------------------------

def gen_interp_video(G, mp4: str, seeds, shuffle_seed=None, w_frames=60*4, kind='cubic', grid_dims=(1,1),
                     num_keyframes=None, wraps=2, psi=1, device=torch.device('cuda'), class_idx=None,
                     rgba=False, rgba_mode='', rgba_mult=1, mask_cutoff=0.4, filt_mode=False,  **video_kwargs):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w*grid_h) != 0:
            raise ValueError('Number of input seeds must be divisible by grid W*H')
        num_keyframes = len(seeds) // (grid_w*grid_h)

    all_seeds = np.zeros(num_keyframes*grid_h*grid_w, dtype=np.int64)
    for idx in range(num_keyframes*grid_h*grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])).to(device).float()
    # Labels.
    label = torch.zeros([zs.size(0), G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    ws = G.mapping(z=zs, c=label, truncation_psi=psi)
    _ = G.synthesis(ws[:1], c=label) # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(mp4, mode='I', fps=60, codec='libx264', **video_kwargs)
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device).float()
                img = G.synthesis(w.unsqueeze(0), c=label, noise_mode='const')[0]

                # check for segmentation task
                if rgba:
                    lo, hi = -1, 1
                    img = (img - lo) * (255 / (hi - lo))

                    # reconstruct the alpha channel as needed
                    if rgba_mode == 'mean_extract':
                        img = img.cpu()
                        # decode the masks, make copies to avoid strange outputs
                        raw_mask = np.empty_like(img[-1:, :, :]).squeeze(0)
                        raw_img = np.empty_like(img[:-1, :, :])
                        # cast values
                        raw_mask[:] = img[-1:, :, :][:]
                        raw_img[:] = img[:-1, :, :][:]

                        # get the binary masks
                        bmask = (raw_mask * 4) - np.sum(raw_img, axis=0)
                        out_mask = np.empty_like(bmask)
                        _, mask_grouping = np.histogram(bmask.flatten(), bins=500)
                        try:
                            mask_k = KMeans(n_clusters=2, random_state=0).fit(mask_grouping.reshape(-1, 1))
                            out_mask[
                                bmask >= mask_grouping[mask_k.labels_ == mask_k.cluster_centers_.argmax()].min()] = 255
                            out_mask[
                                bmask < mask_grouping[mask_k.labels_ == mask_k.cluster_centers_.argmax()].min()] = 0
                            img[-1:, :, :][:] = torch.from_numpy(out_mask)[:]
                        except:
                            print('WARNING: Generated masks cant be clustered... if this isnt init, check generator.')
                            continue

                        # Convert the image
                        img = np.rint(img.permute(1, 2, 0).numpy()).astype('uint8')

                        if filt_mode:
                            # we now need to clean the mask as needed
                            # 1) morphological transformation (eliminate small islands)
                            se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                            se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                            mask = cv2.morphologyEx(img[:, :, -1], cv2.MORPH_CLOSE, se1)
                            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
                            img[..., -1] = mask

                            # 2) flood fill (fill in any holes)
                            # Copy the thresholded image.
                            im_floodfill = img[..., -1].copy()

                            # Mask used to flood filling.
                            # Notice the size needs to be 2 pixels than the image.
                            h, w = img[..., -1].shape[:2]
                            mask = np.zeros((h + 2, w + 2), np.uint8)

                            # Floodfill from point (0, 0)
                            cv2.floodFill(im_floodfill, mask, (0, 0), 255)

                            # Invert floodfilled image
                            im_floodfill_inv = cv2.bitwise_not(im_floodfill)

                            # Combine the two images to get the foreground.
                            img[..., -1] = img[..., -1] | im_floodfill_inv

                            # 3) size filtering (remove any contours that are too small)
                            contour_stack = []
                            contours, _ = cv2.findContours(img[..., -1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            mask = np.zeros_like(img[..., -1])
                            for contour in contours:
                                area = cv2.contourArea(contour)
                                if area > mask_cutoff:
                                    contour_stack.append(contour)
                        else:
                            # get the contours
                            contour_stack = []
                            contours, _ = cv2.findContours(img[..., -1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            mask = np.zeros_like(img[..., -1])
                            for contour in contours:
                                    contour_stack.append(contour)

                        # draw the contours onto the image
                        img = img.copy()[:, :, :-1]
                        img = cv2.drawContours(img.astype(np.uint8), contour_stack, -1, (255, 0, 0), 3)

                        # append
                        img = np.interp(img, (0, 255), (-1, +1))
                        imgs.append(torch.from_numpy(img).permute(2, 0, 1).float().to(device))
                else:
                    imgs.append(img)

        video_out.append_data(layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h))

    video_out.close()

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List[int]]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

def parse_tuple(s: Union[str, Tuple[int,int]]) -> Tuple[int, int]:
    '''Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    m = re.match(r'^(\d+)[x,](\d+)$', s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f'cannot parse tuple {s}')

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/data/pcicales/diffusionGAN/00000-fastgan-GLOM_RGBA_SG2_FULL-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.45-ada_kimg100/best_model.pkl')
@click.option('--seeds', type=parse_range, help='List of random seeds (must be divisible by grid W*H)', default='0-59')
@click.option('--shuffle-seed', type=int, help='Random seed to use for shuffling seed order', default=None)
@click.option('--grid', type=parse_tuple, help='Grid width/height, e.g. \'4x3\' (default: 1x1)', default=(4,3))
@click.option('--num-keyframes', type=int, help='Number of seeds to interpolate through.  If not specified, determine based on the length of the seeds array given by --seeds.', default=None)
@click.option('--w-frames', type=int, help='Number of frames to interpolate between latents', default=120)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--output', help='Output .mp4 filename', type=str, default='/data/public/HULA/GEN_GLOM_RGBA_VIDEOS/{}.mp4', metavar='FILE')
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--force_cpu', help='Force cpu usage even if using CUDA', type=bool, default=True, metavar='BOOL')


# Segmentation config
@click.option('--rgba',       help='Whether or not we are generating with mask', metavar='BOOL', type=bool, default=True)
@click.option('--rgba_mode',  help='How we encode our masks', metavar='STR', type=click.Choice(['mean_extract', 'naive']), default='mean_extract')
@click.option('--rgba_mult',  help='What multiplier to use on binary mask', metavar='INT', type=int, default=3)
@click.option('--mask_cutoff',  help='Pixel filtering to remove noisy contours', metavar='INT', type=int, default=200)
@click.option('--filt_mode',       help='Whether or not we are using cv2 filtering', metavar='BOOL', type=bool, default=False)


def generate_images(
    network_pkl: str,
    seeds: List[int],
    shuffle_seed: Optional[int],
    truncation_psi: float,
    grid: Tuple[int,int],
    num_keyframes: Optional[int],
    w_frames: int,
    output: str,
    force_cpu: bool,
    filt_mode: bool,
    class_idx: Optional[int],
    rgba: Optional[bool],
    rgba_mode: str,
    rgba_mult: Optional[int],
    mask_cutoff: Optional[int]
):
    """Render a latent vector interpolation video.

    Examples:

    \b
    # Render a 4x2 grid of interpolations for seeds 0 through 31.
    python gen_video.py --output=lerp.mp4 --trunc=1 --seeds=0-31 --grid=4x2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    Animation length and seed keyframes:

    The animation length is either determined based on the --seeds value or explicitly
    specified using the --num-keyframes option.

    When num keyframes is specified with --num-keyframes, the output video length
    will be 'num_keyframes*w_frames' frames.

    If --num-keyframes is not specified, the number of seeds given with
    --seeds must be divisible by grid size W*H (--grid).  In this case the
    output video length will be '# seeds/(w*h)*w_frames' frames.
    """

    print('Loading networks from "%s"...' % network_pkl)
    if force_cpu:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if rgba:
        if '{}' not in output:
            raise AssertionError('Output must have `{}` in file name to identify the video!')

        if '/' in output:
            os.makedirs(output.split('{}', 1)[0], exist_ok=True)

        output = output.format(network_pkl.split('/')[-2])

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    gen_interp_video(G=G, mp4=output, bitrate='12M', grid_dims=grid, num_keyframes=num_keyframes, w_frames=w_frames,
                     seeds=seeds, shuffle_seed=shuffle_seed, psi=truncation_psi, device=device, class_idx=class_idx, rgba=rgba, rgba_mode=rgba_mode,
                     rgba_mult=rgba_mult, mask_cutoff=mask_cutoff, filt_mode=filt_mode)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
