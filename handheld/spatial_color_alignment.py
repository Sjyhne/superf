# Copyright (c) 2021 Huawei Technologies Co., Ltd.
# Licensed under CC BY-NC-SA 4.0 (Attribution-NonCommercial-ShareAlike 4.0 International) (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode
#
# The code is released for academic research use only. For commercial use, please contact Huawei Technologies Co., Ltd.
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def warp(feat, flow, mode='bilinear', padding_mode='zeros'):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow im1 --> im2

    input flow must be in format (x, y) at every pixel
    feat: [B, C, H, W] (im2)
    flow: [B, 2, H, W] flow (x, y)

    """
    B, C, H, W = feat.size()

    # mesh grid
    rowv, colv = torch.meshgrid([torch.arange(0.5, H + 0.5, device=feat.device),
                                 torch.arange(0.5, W + 0.5, device=feat.device)])
    grid = torch.stack((colv, rowv), dim=0).unsqueeze(0).float()
    grid = grid + flow

    # scale grid to [-1,1]
    grid_norm_c = 2.0 * grid[:, 0] / W - 1.0
    grid_norm_r = 2.0 * grid[:, 1] / H - 1.0

    grid_norm = torch.stack((grid_norm_c, grid_norm_r), dim=1)

    grid_norm = grid_norm.permute(0, 2, 3, 1)

    output = F.grid_sample(feat, grid_norm, mode=mode, padding_mode=padding_mode)

    return output

def gauss_1d(sz, sigma, center, end_pad=0, density=False):
    """ Returns a 1-D Gaussian """
    k = torch.arange(-(sz-1)/2, (sz+1)/2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0/(2*sigma**2) * (k - center.reshape(-1, 1))**2)
    if density:
        gauss /= math.sqrt(2*math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False):
    """ Returns a 2-D Gaussian """
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    if isinstance(sz, int):
        sz = (sz, sz)

    if isinstance(center, (list, tuple)):
        center = torch.tensor(center).view(1, 2)

    return gauss_1d(sz[0], sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1], sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)



def get_gaussian_kernel(sd, ksz=None):
    """ Returns a 2D Gaussian kernel with standard deviation sd """
    if ksz is None:
        ksz = int(4 * sd + 1)

    assert ksz % 2 == 1
    K = gauss_2d(ksz, sd, (0.0, 0.0), density=True)
    K = K / K.sum()
    return K.unsqueeze(0), ksz


def apply_kernel(im, ksz, kernel):
    """ apply the provided kernel on input image """
    shape = im.shape
    im = im.view(-1, 1, *im.shape[-2:])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_out = F.conv2d(im, kernel).view(shape)
    return im_out


def match_colors(im_ref, im_q, im_test, ksz, gauss_kernel):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    im_ref_mean_re = im_ref_mean.view(*im_ref_mean.shape[:2], -1)
    im_q_mean_re = im_q_mean.view(*im_q_mean.shape[:2], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.lstsq(ir.t(), iq.t())
        c = c.solution[:3]
        c_mat_all.append(c)

    c_mat = torch.stack(c_mat_all, dim=0)
    im_q_mean_conv = torch.matmul(im_q_mean_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_q_mean_conv = im_q_mean_conv.view(im_q_mean.shape)

    err = ((im_q_mean_conv - im_ref_mean) * 255.0).norm(dim=1)

    thresh = 20

    # If error is larger than a threshold, ignore these pixels
    valid = err < thresh

    pad = (im_q.shape[-1] - valid.shape[-1]) // 2
    pad = [pad, pad, pad, pad]
    valid = F.pad(valid, pad)

    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    valid = valid > 0.9

    # Apply the transformation to test image
    im_test_re = im_test.view(*im_test.shape[:2], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    return im_t_conv, valid


class SpatialColorAlignment(nn.Module):
    def __init__(self, alignment_net, sr_factor=4):
        super().__init__()
        self.sr_factor = sr_factor
        self.alignment_net = alignment_net

        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.alignment_net.to(device)
        self.gauss_kernel = self.gauss_kernel.to(device)

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = warp.warp(pred, flow)

        # Warp the base input frame to the ground truth. This will be used to estimate the color transformation between
        # the input and the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(2.0 * sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        burst_0 = burst_input[:, 0, [0, 1, 3]].contiguous()
        burst_0_warped = warp.warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz, self.gauss_kernel)

        return pred_warped_m, valid 