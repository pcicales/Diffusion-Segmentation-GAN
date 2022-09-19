# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import cv2

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

import legacy

#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
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

def parse_vec2(s: Union[str, Tuple[float, float]]) -> Tuple[float, float]:
    '''Parse a floating point 2-vector of syntax 'a,b'.

    Example:
        '0,1' returns (0,1)
    '''
    if isinstance(s, tuple): return s
    parts = s.split(',')
    if len(parts) == 2:
        return (float(parts[0]), float(parts[1]))
    raise ValueError(f'cannot parse 2-vector {s}')

#----------------------------------------------------------------------------

def make_transform(translate: Tuple[float,float], angle: float):
    m = np.eye(3)
    s = np.sin(angle/360.0*np.pi*2)
    c = np.cos(angle/360.0*np.pi*2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', default='/data/pcicales/diffusionGAN/00000-fastgan-GLOM_RGBA_SG2_FULL-gpus2-batch32-d_pos-first-noise_sd-0.5-target0.45-ada_kimg100/best_model.pkl')
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', default='0-500')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--translate', help='Translate XY-coordinate (e.g. \'0.3,1\')', type=parse_vec2, default='0,0', show_default=True, metavar='VEC2')
@click.option('--rotate', help='Rotation angle in degrees', type=float, default=0, show_default=True, metavar='ANGLE')
@click.option('--outdir', help='Where to save the output images', type=str, default='/data/public/HULA/GEN_GLOM_RGBA', metavar='DIR')

# Segmentation config
@click.option('--rgba',       help='Whether or not we are generating with mask', metavar='BOOL', type=bool, default=True)
@click.option('--rgba_mode',  help='How we encode our masks', metavar='STR', type=click.Choice(['mean_extract', 'naive']), default='mean_extract')
@click.option('--rgba_mult',  help='What multiplier to use on binary mask', metavar='INT', type=int, default=3)
@click.option('--mask_cutoff',  help='Pixel filtering to remove noisy contours', metavar='INT', type=int, default=200)
@click.option('--mask_filter',  help='Percent of image that must be a mask to keep the sample', metavar='FLOAT', type=float, default=0.2)
@click.option('--discard_outdir', help='Where to save the discarded output images, set to None if you wish to discard them', type=str, default='/data/public/HULA/GEN_GLOM_RGBA_DISCARD', metavar='DIR')
@click.option('--filt_mode',       help='Whether or not we are using cv2 filtering', metavar='BOOL', type=bool, default=False)

def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    translate: Tuple[float,float],
    rotate: float,
    class_idx: Optional[int],
    rgba: Optional[bool],
    rgba_mode: str,
    rgba_mult: Optional[int],
    filt_mode: Optional[bool],
    mask_cutoff: Optional[int],
    mask_filter: Optional[float],
    discard_outdir: str,
):
    """Generate images using pretrained network pickle.

    Examples:

    \b--
    # Generate an image using pre-trained AFHQv2 model ("Ours" in Figure 1, left).
    python gen_images.py --outdir=out --trunc=1 --seeds=2 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-r-afhqv2-512x512.pkl

    \b
    # Generate uncurated images with truncation using the MetFaces-U dataset
    python gen_images.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/stylegan3-t-metfacesu-1024x1024.pkl
    """

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    os.makedirs(outdir, exist_ok=True)
    if discard_outdir != None:
        os.makedirs(discard_outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        if class_idx is None:
            raise click.ClickException('Must specify class label with --class when using a conditional network')
        label[:, class_idx] = 1
    else:
        if class_idx is not None:
            print ('warn: --class=lbl ignored when running on an unconditional network')

    # Generate images.
    for seed_idx, seed in enumerate(seeds):
        print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device).float()

        # Construct an inverse rotation/translation matrix and pass to the generator.  The
        # generator expects this matrix as an inverse to avoid potentially failing numerical
        # operations in the network.
        if hasattr(G.synthesis, 'input'):
            m = make_transform(translate, rotate)
            m = np.linalg.inv(m)
            G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)

        if rgba:
            lo, hi = -1, 1
            img = (img - lo) * (255 / (hi - lo))

            # reconstruct the alpha channel as needed
            if rgba_mode == 'mean_extract':
                img = img.cpu().squeeze(0)
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
                    out_mask[bmask >= mask_grouping[mask_k.labels_ == mask_k.cluster_centers_.argmax()].min()] = 255
                    out_mask[bmask < mask_grouping[mask_k.labels_ == mask_k.cluster_centers_.argmax()].min()] = 0
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
                    contours, _ = cv2.findContours(img[..., -1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    mask = np.zeros_like(img[..., -1])
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > mask_cutoff:
                            cv2.fillPoly(mask, [contour], 255)
                    img[..., -1][:] = mask[:]

                # 4) filter by mask size (remove images that dont have an acceptably large mask)
                mask_area = ((np.sum(img[..., -1]) / 255) / (img.shape[0] * img.shape[1]))
                if mask_area < mask_filter:
                    print('Small mask detected for seed{0}, mask occupied {1:.02%}% of the image.'.format(seed, mask_area))
                    if discard_outdir != None:
                        PIL.Image.fromarray(img, 'RGBA').save(f'{discard_outdir}/seed{seed:04d}.png')
                    continue

                # save the image as RGBA
                PIL.Image.fromarray(img, 'RGBA').save(f'{outdir}/seed{seed:04d}.png')

        else:
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/seed{seed:04d}.png')


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
