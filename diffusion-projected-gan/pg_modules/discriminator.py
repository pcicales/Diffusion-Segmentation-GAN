from functools import partial
import numpy as np
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
from pg_modules.projector import F_RandomProj
from pg_modules.diffaug import DiffAugment

import random


class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)


class SingleDiscCond(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False, c_dim=1000, cmap_dim=64, embedding_dim=128):
        super().__init__()
        self.cmap_dim = cmap_dim

        # midas channels
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2
        self.main = nn.Sequential(*layers)

        # additions for conditioning on class information
        self.cls = conv2d(nfc[end_sz], self.cmap_dim, 4, 1, 0, bias=False)
        self.embed = nn.Embedding(num_embeddings=c_dim, embedding_dim=embedding_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(self.embed.embedding_dim, self.cmap_dim),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, c):
        h = self.main(x)
        out = self.cls(h)

        # conditioning via projection
        cmap = self.embed_proj(self.embed(c.argmax(1))).unsqueeze(-1).unsqueeze(-1)
        out = (out * cmap).sum(dim=1, keepdim=True) * (1 / np.sqrt(self.cmap_dim))

        return out


class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        num_discs=1,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert num_discs in [1, 2, 3, 4]

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = channels[:num_discs]
        self.disc_in_res = resolutions[:num_discs]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            all_logits.append(disc(features[k], c).view(features[k].size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits


class ProjectedDiscriminator(torch.nn.Module):
    def __init__(
        self,
        diffaug=True,
        interp224=True,
        rgba=False,
        rgba_mode='',
        multi_disc=False,
        imnet_norm=False,
        channel_inc = 0,
        backbone_kwargs={},
        **kwargs
    ):
        super().__init__()
        self.diffaug = diffaug
        self.rgba = rgba
        self.rgba_mode = rgba_mode
        self.multi_disc = multi_disc
        self.imnet_norm = imnet_norm
        self.channel_inc = channel_inc
        self.interp224 = interp224

        # normalization implementation
        if self.imnet_norm:
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
        else:
            self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5])

        # progressive channel addition
        if self.channel_inc > 0:
            self.current_mode = 1
            if self.rgba:
                self.repeat_chan = 4
            else:
                self.repeat_chan = 3

        self.feature_network = F_RandomProj(rgba=rgba, rgba_mode=rgba_mode, multi_disc=multi_disc, disc_noise=kwargs['disc_noise'], **backbone_kwargs)
        if self.rgba and self.multi_disc:
            self.discriminator = MultiScaleD(
                channels=self.feature_network.CHANNELS,
                resolutions=self.feature_network.RESOLUTIONS,
                **backbone_kwargs,
            )
            self.mask_discriminator = MultiScaleD(
                channels=self.feature_network.CHANNELS,
                resolutions=self.feature_network.RESOLUTIONS,
                **backbone_kwargs,
            )
        elif self.rgba and not self.multi_disc:
            self.discriminator = MultiScaleD(
                channels=[chan * 2 for chan in self.feature_network.CHANNELS],
                resolutions=self.feature_network.RESOLUTIONS,
                **backbone_kwargs,
            )
        else:
            self.discriminator = MultiScaleD(
                channels=self.feature_network.CHANNELS,
                resolutions=self.feature_network.RESOLUTIONS,
                **backbone_kwargs,
            )

    def train(self, mode=True):
        self.feature_network = self.feature_network.train(False)
        self.discriminator = self.discriminator.train(mode)
        if self.rgba and self.multi_disc:
            self.mask_discriminator = self.mask_discriminator.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x, c, real_in):

        # progressive channel addition
        if self.channel_inc > 0:
            if ((self.current_mode < 3) and not self.rgba) or ((self.current_mode < 4) and self.rgba):
                temp_x = torch.empty_like(x)
                if real_in:
                    temp_x[:] = x[:, self.current_mode-1, :, :].unsqueeze(1).expand(x.shape)[:]
                else:
                    samp_batch = torch.tensor([random.randint(self.current_mode-1, self.repeat_chan-1) for _ in range(x.shape[0])], device=x.device)
                    temp_x[:] = x[torch.arange(x.shape[0]), samp_batch].unsqueeze(1).expand(x.shape)[:]
                # R is already done initially
                # RG
                if self.current_mode == 2:
                    temp_x[:, 0, ...][:] = x[:, 0, ...][:]
                # RGB (only for RGBA, else its just the input)
                elif self.current_mode == 3:
                    temp_x[:, 0:2, ...][:] = x[:, 0:2, ...][:]
                # RGBA is the initial input
                x = temp_x

        # if we are using multi disc, we need to get the outputs for both discriminators
        if self.rgba and self.multi_disc:
            emask = x[:, -1:, ...].expand(x.shape[0], 3, x.shape[2], x.shape[3])
            x = x[:, :-1, ...]
            base_batch = x.shape[0]
            if self.diffaug:
                x = DiffAugment(x, policy='color,translation,cutout')
                emask = DiffAugment(emask, policy='color,translation,cutout')

            if self.interp224:
                x = F.interpolate(x, 224, mode='bilinear', align_corners=False)
                emask = F.interpolate(emask, 224, mode='bilinear', align_corners=False) # fix

            # normalize after augmentation
            x = self.normalize(x)
            emask = self.normalize(emask)

            # make a cat tensor to speed up feature extraction
            x = torch.cat((x, emask), dim=0)
            features = self.feature_network(x)
            mask_features = {key: out_feat[base_batch:, ...] for key, out_feat in zip(features.keys(), features.values())}
            img_features = {key: out_feat[:base_batch, ...] for key, out_feat in zip(features.keys(), features.values())}
            logits = self.discriminator(img_features, c)
            mask_logits = self.mask_discriminator(mask_features, c)

            # cat logits, may need to keep these separate after testing
            logits = torch.cat((logits, mask_logits), dim=1)

            return logits

        elif self.rgba and not self.multi_disc:
            if self.diffaug:
                x = DiffAugment(x, policy='color,translation,cutout')

            if self.interp224:
                x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

            emask = x[:, -1:, ...].expand(x.shape[0], 3, x.shape[2], x.shape[3])
            x = x[:, :-1, ...]

            # normalize after augmentation
            x = self.normalize(x)
            emask = self.normalize(emask)

            # make a cat tensor to speed up feature extraction
            base_batch = x.shape[0]
            x = torch.cat((x, emask), dim=0)
            features = self.feature_network(x)

            # make the appended feature dict to pass to our discriminator
            features = {key: torch.cat(featraw.split(base_batch, dim=0), dim=1) for key, featraw in zip(features.keys(), features.values())}

            logits = self.discriminator(features, c)

            return logits

        else:
            if self.diffaug:
                x = DiffAugment(x, policy='color,translation,cutout')

            if self.interp224:
                x = F.interpolate(x, 224, mode='bilinear', align_corners=False)

            # normalize after augmentation
            x = self.normalize(x)

            features = self.feature_network(x)
            logits = self.discriminator(features, c)

            return logits