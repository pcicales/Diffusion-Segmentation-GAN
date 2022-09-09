import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from pg_modules.blocks import FeatureFusionBlock
from pg_modules.diffusion import Diffusion


def _make_scratch_ccm(scratch, in_channels, cout, expand=False):
    # shapes
    out_channels = [cout, cout*2, cout*4, cout*8] if expand else [cout]*4

    scratch.layer0_ccm = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer1_ccm = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer2_ccm = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=True)
    scratch.layer3_ccm = nn.Conv2d(in_channels[3], out_channels[3], kernel_size=1, stride=1, padding=0, bias=True)

    scratch.CHANNELS = out_channels

    return scratch


def _make_scratch_csm(scratch, in_channels, cout, expand):
    scratch.layer3_csm = FeatureFusionBlock(in_channels[3], nn.ReLU(False), expand=expand, lowest=True)
    scratch.layer2_csm = FeatureFusionBlock(in_channels[2], nn.ReLU(False), expand=expand)
    scratch.layer1_csm = FeatureFusionBlock(in_channels[1], nn.ReLU(False), expand=expand)
    scratch.layer0_csm = FeatureFusionBlock(in_channels[0], nn.ReLU(False))

    # last refinenet does not expand to save channels in higher dimensions
    scratch.CHANNELS = [cout, cout, cout*2, cout*4] if expand else [cout]*4

    return scratch


def _make_efficientnet(model):
    pretrained = nn.Module()
    # perhaps due to an older version of timm...
    # pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained


def calc_channels(pretrained, inp_res=224, rgba=False, rgba_mode=''):
    channels = []
    if rgba:
        tmp = torch.zeros(1, 4, inp_res, inp_res)
    else:
        tmp = torch.zeros(1, 3, inp_res, inp_res)

    # forward pass
    tmp = pretrained.layer0(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer1(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer2(tmp)
    channels.append(tmp.shape[1])
    tmp = pretrained.layer3(tmp)
    channels.append(tmp.shape[1])

    return channels


def _make_projector(im_res, cout, proj_type, expand=False, rgba=False, rgba_mode=''):
    assert proj_type in [0, 1, 2], "Invalid projection type"

    ### Build pretrained feature network
    model = timm.create_model('tf_efficientnet_lite0', pretrained=True)
    if rgba:
        # change to 4 channel input
        model.conv_stem.in_channels = 4
        # experimenting with assigning a pretrained weight to the fourth dim...
        model.conv_stem.weight = torch.nn.Parameter(torch.cat((model.conv_stem.weight, model.conv_stem.weight[:, :-2, :, :]), dim=1))
    pretrained = _make_efficientnet(model)

    # determine resolution of feature maps, this is later used to calculate the number
    # of down blocks in the discriminators. Interestingly, the best results are achieved
    # by fixing this to 256, ie., we use the same number of down blocks per discriminator
    # independent of the dataset resolution
    im_res = 256
    pretrained.RESOLUTIONS = [im_res//4, im_res//8, im_res//16, im_res//32]
    pretrained.CHANNELS = calc_channels(pretrained, rgba=rgba, rgba_mode=rgba_mode)

    if proj_type == 0: return pretrained, None

    ### Build CCM
    scratch = nn.Module()
    scratch = _make_scratch_ccm(scratch, in_channels=pretrained.CHANNELS, cout=cout, expand=expand)
    pretrained.CHANNELS = scratch.CHANNELS

    if proj_type == 1: return pretrained, scratch

    ### build CSM
    scratch = _make_scratch_csm(scratch, in_channels=scratch.CHANNELS, cout=cout, expand=expand)

    # CSM upsamples x2 so the feature map resolution doubles
    pretrained.RESOLUTIONS = [res*2 for res in pretrained.RESOLUTIONS]
    pretrained.CHANNELS = scratch.CHANNELS

    return pretrained, scratch


def rescale(out):
    out_min, out_max = out.min(), out.max()
    return (out - out_min) / (out_max - out_min) * 2 - 1


class F_RandomProj(nn.Module):
    def __init__(
        self,
        im_res=256,
        cout=64,
        expand=True,
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        d_pos='first',
        noise_sd=0.5,
        rgba=False,
        rgba_mode='',
        **kwargs,
    ):
        super().__init__()
        self.proj_type = proj_type
        self.cout = cout
        self.expand = expand
        self.rgba = rgba
        self.rgba_mode = rgba_mode

        self.d_pos = d_pos
        self.noise_sd = noise_sd
        # self.diffusion = AugmentPipe(t_max=1000)
        self.diffusion = Diffusion(t_min=5, t_max=500, beta_start=1e-4, beta_end=1e-2)
        # build pretrained feature network and random decoder (scratch)
        self.pretrained, self.scratch = _make_projector(im_res=im_res, cout=self.cout, proj_type=self.proj_type, expand=self.expand, rgba=rgba, rgba_mode=rgba_mode)
        self.CHANNELS = self.pretrained.CHANNELS
        self.RESOLUTIONS = self.pretrained.RESOLUTIONS

    def forward(self, x):
        # x = self.diffusion(x, noise_std=0.05)
        # predict feature maps
        out0 = self.pretrained.layer0(x)
        out1 = self.pretrained.layer1(out0)
        out2 = self.pretrained.layer2(out1)
        out3 = self.pretrained.layer3(out2)

        # start enumerating at the lowest layer (this is where we put the first discriminator)
        out = {
            '0': out0,
            '1': out1,
            '2': out2,
            '3': out3,
        }

        if self.d_pos == 'first':
            out['0'] = self.diffusion(out['0'], noise_std=self.noise_sd)
            out['1'] = self.diffusion(out['1'], noise_std=self.noise_sd)
            out['2'] = self.diffusion(out['2'], noise_std=self.noise_sd)
            out['3'] = self.diffusion(out['3'], noise_std=self.noise_sd)

        if self.proj_type == 0: return out

        out0_channel_mixed = self.scratch.layer0_ccm(out['0'])
        out1_channel_mixed = self.scratch.layer1_ccm(out['1'])
        out2_channel_mixed = self.scratch.layer2_ccm(out['2'])
        out3_channel_mixed = self.scratch.layer3_ccm(out['3'])

        out = {
            '0': out0_channel_mixed,
            '1': out1_channel_mixed,
            '2': out2_channel_mixed,
            '3': out3_channel_mixed,
        }

        if self.proj_type == 1: return out

        # from bottom to top
        out3_scale_mixed = self.scratch.layer3_csm(out3_channel_mixed)
        out2_scale_mixed = self.scratch.layer2_csm(out3_scale_mixed, out2_channel_mixed)
        out1_scale_mixed = self.scratch.layer1_csm(out2_scale_mixed, out1_channel_mixed)
        out0_scale_mixed = self.scratch.layer0_csm(out1_scale_mixed, out0_channel_mixed)

        out = {
            '0': out0_scale_mixed,
            '1': out1_scale_mixed,
            '2': out2_scale_mixed,
            '3': out3_scale_mixed,
        }

        if self.d_pos == 'last':
            out['0'] = self.diffusion(out['0'], noise_std=self.noise_sd)
            out['1'] = self.diffusion(out['1'], noise_std=self.noise_sd)
            out['2'] = self.diffusion(out['2'], noise_std=self.noise_sd)
            out['3'] = self.diffusion(out['3'], noise_std=self.noise_sd)
        # CDA
        # n_sd1, n_sd2 = 0.5, 0.25
        # n_sd1, n_sd2 = 0.25, 0.1
        # out['0'], t0 = self.diffusion(out['0'], noise_std=n_sd1)
        # out['1'], t1 = self.diffusion(out['1'], noise_std=n_sd1)
        # out['2'], t2 = self.diffusion(out['2'], noise_std=n_sd2)
        # out['3'], t3 = self.diffusion(out['3'], noise_std=n_sd2)
        # diffusion_t = {'0': t0, '1': t1, '2': t2, '3': t3}

        return out
