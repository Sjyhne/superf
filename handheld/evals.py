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

import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

import numpy as np
import math
from math import exp  # Add explicit import for exp function

from utils import align_output_to_target, align_kornia_brute_force

# Code from https://github.com/jorge-pessoa/pytorch-msssim  licensed under MIT license
# https://github.com/jorge-pessoa/pytorch-msssim/blob/master/LICENSE.txt


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
    im = im.view(-1, 1, shape[-2], shape[-1])

    pad = [ksz // 2, ksz // 2, ksz // 2, ksz // 2]
    im = F.pad(im, pad, mode='reflect')
    im_out = F.conv2d(im, kernel).view(shape)
    return im_out

def lispr_warp(feat, flow, mode='bilinear', padding_mode='zeros'):
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


def match_colors(im_ref, im_q, im_test, ksz, gauss_kernel):
    """ Estimates a color transformation matrix between im_ref and im_q. Applies the estimated transformation to
        im_test
    """
    gauss_kernel = gauss_kernel.to(im_ref.device)
    bi = 5

    print(f"Input shapes - im_ref: {im_ref.shape}, im_q: {im_q.shape}, im_test: {im_test.shape}")
    
    # Determine the target size - we'll work in higher resolution for better results
    # Get dimensions from the highest resolution tensor
    max_height = max(im_ref.shape[2], im_q.shape[2], im_test.shape[2])
    max_width = max(im_ref.shape[3], im_q.shape[3], im_test.shape[3])
    target_size = (max_height, max_width)
    
    # Resize images to the target size
    if im_ref.shape[2:] != target_size:
        print(f"Upsampling im_ref from {im_ref.shape} to size {target_size}")
        im_ref = F.interpolate(im_ref, size=target_size, mode='bilinear', align_corners=False)
    
    if im_q.shape[2:] != target_size:
        print(f"Upsampling im_q from {im_q.shape} to size {target_size}")
        im_q = F.interpolate(im_q, size=target_size, mode='bilinear', align_corners=False)
    
    if im_test.shape[2:] != target_size:
        print(f"Upsampling im_test from {im_test.shape} to size {target_size}")
        im_test = F.interpolate(im_test, size=target_size, mode='bilinear', align_corners=False)
    
    print(f"After resize - im_ref: {im_ref.shape}, im_q: {im_q.shape}, im_test: {im_test.shape}")

    # Apply Gaussian smoothing
    im_ref_mean = apply_kernel(im_ref, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()
    im_q_mean = apply_kernel(im_q, ksz, gauss_kernel)[:, :, bi:-bi, bi:-bi].contiguous()

    shape_ref = im_ref_mean.shape
    shape_q = im_q_mean.shape
    im_ref_mean_re = im_ref_mean.view(shape_ref[0], shape_ref[1], -1)
    im_q_mean_re = im_q_mean.view(shape_q[0], shape_q[1], -1)

    # Estimate color transformation matrix by minimizing the least squares error
    c_mat_all = []
    for ir, iq in zip(im_ref_mean_re, im_q_mean_re):
        c = torch.linalg.lstsq(ir.t(), iq.t())
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

    # Apply the transformation to test image
    shape_test = im_test.shape
    im_test_re = im_test.view(shape_test[0], shape_test[1], -1)
    im_t_conv = torch.matmul(im_test_re.permute(0, 2, 1), c_mat).permute(0, 2, 1)
    im_t_conv = im_t_conv.view(im_test.shape)

    # Calculate upscale factor for valid mask
    upsample_factor = im_test.shape[-1] / valid.shape[-1]
    valid = F.interpolate(valid.unsqueeze(1).float(), scale_factor=upsample_factor, mode='bilinear')
    valid = valid > 0.9

    return im_t_conv, valid


class SpatialColorAlignment(nn.Module):
    def __init__(self, alignment_net, sr_factor=4):
        super().__init__()
        self.alignment_net = alignment_net
        self.sr_factor = sr_factor
        self.gauss_kernel, self.ksz = get_gaussian_kernel(sd=1.5)

    def to(self, device):
        """ Move the network to device
        args:
            device - device to use. 'cpu' or 'cuda'
        """
        self.alignment_net.to(device)
        self.gauss_kernel = self.gauss_kernel.to(device)
        return self

    def forward(self, pred, gt, burst_input):
        # Estimate flow between the prediction and the ground truth
        with torch.no_grad():
            flow = self.alignment_net(pred / (pred.max() + 1e-6), gt / (gt.max() + 1e-6))

        # Warp the prediction to the ground truth coordinates
        pred_warped = lispr_warp(pred, flow)

        # Warp the base input frame to the ground truth
        sr_factor = self.sr_factor
        ds_factor = 1.0 / float(sr_factor)
        flow_ds = F.interpolate(flow, scale_factor=ds_factor, mode='bilinear') * ds_factor

        burst_0 = burst_input


        burst_0_warped = lispr_warp(burst_0, flow_ds)
        frame_gt_ds = F.interpolate(gt, scale_factor=ds_factor, mode='bilinear')

        # Match the colorspace between the prediction and ground truth
        pred_warped_m, valid = match_colors(frame_gt_ds, burst_0_warped, pred_warped, self.ksz, self.gauss_kernel)

        if use_custom_alignment:
            try:
                pred_warped = align_output_to_target(pred, gt, spatial=False, spectral=True)
                aligned = align_kornia_brute_force(pred.squeeze(0), gt.squeeze(0)).unsqueeze(0)
                return aligned, torch.ones_like(aligned[:, 0:1])
            except Exception as e:
                print(f"Custom alignment failed: {e}, falling back to default")
                return pred_warped_m, valid
        else:
            return pred_warped_m, valid


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None, spatial_out=False):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    window = window.to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if spatial_out:
        ret = ssim_map
    elif size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret


def msssim(img1, img2, window_size=11, size_average=True, val_range=None, normalize=False):
    device = img1.device
    weights = torch.FloatTensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(device)
    levels = weights.size()[0]
    mssim = []
    mcs = []
    for _ in range(levels):
        sim, cs = ssim(img1, img2, window_size=window_size, size_average=size_average, full=True, val_range=val_range)
        mssim.append(sim)
        mcs.append(cs)

        img1 = F.avg_pool2d(img1, (2, 2))
        img2 = F.avg_pool2d(img2, (2, 2))

    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)

    # Normalize (to avoid NaNs during training unstable models, not compliant with original definition)
    if normalize:
        mssim = (mssim + 1) / 2
        mcs = (mcs + 1) / 2

    pow1 = mcs ** weights
    pow2 = mssim ** weights
    # From Matlab implementation https://ece.uwaterloo.ca/~z70wang/research/iwssim/
    output = torch.prod(pow1[:-1] * pow2[-1])
    return output


# Classes to re-use window
class msssim_SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None, spatial_out=False):
        super(msssim_SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range
        self.spatial_out = spatial_out

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        window = window.to(img1.device)
        return ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average,
                    spatial_out=self.spatial_out)

class MSSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, channel=3):
        super(MSSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel

    def forward(self, img1, img2):
        return msssim(img1, img2, window_size=self.window_size, size_average=self.size_average)


class PixelWiseError(nn.Module):
    """ Computes pixel-wise error using the specified metric. Optionally boundary pixels are ignored during error
        calculation """
    def __init__(self, metric='l1', boundary_ignore=100):
        super().__init__()
        self.boundary_ignore = boundary_ignore

        if metric == 'l1':
            self.loss_fn = F.l1_loss
        elif metric == 'l2':
            self.loss_fn = F.mse_loss
        elif metric == 'l2_sqrt':
            def l2_sqrt(pred, gt):
                return (((pred - gt) ** 2).sum(dim=-3)).sqrt().mean()
            self.loss_fn = l2_sqrt
        elif metric == 'charbonnier':
            def charbonnier(pred, gt):
                eps = 1e-3
                return ((pred - gt) ** 2 + eps**2).sqrt().mean()
            self.loss_fn = charbonnier
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        valid = None
        if self.boundary_ignore is not None:
            # Remove boundary pixels
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            if valid is not None:
                valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Valid indicates image regions which should be used for loss calculation
        if valid is None:
            err = self.loss_fn(pred, gt)
        else:
            err = self.loss_fn(pred, gt, reduction='none')

            eps = 1e-12
            elem_ratio = err.numel() / valid.numel()
            err = (err * valid.float()).sum() / (valid.float().sum() * elem_ratio + eps)

        return err


class MappedLoss(nn.Module):
    def __init__(self, base_loss, mapping_fn=None):
        super().__init__()
        self.base_loss = base_loss
        self.mapping_fn = mapping_fn

    def forward(self, pred, gt, meta_info=None, valid=None):
        if self.mapping_fn is not None:
            pred_l = [self.mapping_fn(p, m) for p, m in zip(pred, meta_info)]
            gt_l = [self.mapping_fn(p, m) for p, m in zip(gt, meta_info)]
            pred = torch.stack(pred_l)
            gt = torch.stack(gt_l)

        err = self.base_loss(pred, gt, valid)
        return err


class PSNR(nn.Module):
    def __init__(self, boundary_ignore=None, max_value=1.0):
        super().__init__()
        self.l2 = PixelWiseError(metric='l2', boundary_ignore=boundary_ignore)
        self.max_value = max_value

    def psnr(self, pred, gt, valid=None):
        mse = self.l2(pred, gt, valid=valid)

        if getattr(self, 'max_value', 1.0) is not None:
            psnr = 20 * math.log10(getattr(self, 'max_value', 1.0)) - 10.0 * torch.log10(torch.tensor(mse))
        else:
            psnr = 20 * gt.max().log10() - 10.0 * torch.log10(torch.tensor(mse))

        if torch.isinf(psnr) or torch.isnan(psnr):
            print('invalid psnr')

        return psnr

    def forward(self, pred, gt, valid=None):
        if valid is None:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0)) for p, g in
                        zip(pred, gt)]
        else:
            psnr_all = [self.psnr(p.unsqueeze(0), g.unsqueeze(0), v.unsqueeze(0)) for p, g, v in zip(pred, gt, valid)]

        psnr_all = [p for p in psnr_all if not (torch.isinf(p) or torch.isnan(p))]

        if len(psnr_all) == 0:
            psnr = 0
        else:
            psnr = sum(psnr_all) / len(psnr_all)
        return psnr


class SSIM(nn.Module):
    """
    SSIM (Structural Similarity Index) metric implementation
    """
    def __init__(self, boundary_ignore=None, max_value=1.0, use_for_loss=True):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.max_value = max_value
        self.use_for_loss = use_for_loss
        self.gaussian_kernel, self.kernel_size = get_gaussian_kernel(sd=1.5)

    def forward(self, pred, target, valid=None):
        """
        Args:
            pred: Prediction tensor
            target: Target tensor
            valid: Optional mask of valid pixels
            
        Returns:
            SSIM value
        """
        if valid is None:
            valid = torch.ones_like(pred[:, 0:1, ...])
        
        if self.boundary_ignore is not None:
            valid_boundary = valid.clone()
            valid_boundary[:, :, :self.boundary_ignore, :] = 0
            valid_boundary[:, :, -self.boundary_ignore:, :] = 0
            valid_boundary[:, :, :, :self.boundary_ignore] = 0
            valid_boundary[:, :, :, -self.boundary_ignore:] = 0
            valid = valid_boundary
        
        # Apply gaussian filter to input and target
        self.gaussian_kernel = self.gaussian_kernel.to(pred.device)
        C1 = (0.01 * self.max_value) ** 2
        C2 = (0.03 * self.max_value) ** 2
        
        mu1 = apply_kernel(pred, self.kernel_size, self.gaussian_kernel)
        mu2 = apply_kernel(target, self.kernel_size, self.gaussian_kernel)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = apply_kernel(pred ** 2, self.kernel_size, self.gaussian_kernel) - mu1_sq
        sigma2_sq = apply_kernel(target ** 2, self.kernel_size, self.gaussian_kernel) - mu2_sq
        sigma12 = apply_kernel(pred * target, self.kernel_size, self.gaussian_kernel) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        # Apply valid mask
        ssim_map = ssim_map * valid
        
        # Calculate mean SSIM
        ssim = ssim_map.sum() / (valid.sum() + 1e-8)
        
        if self.use_for_loss:
            return 1.0 - ssim
        else:
            return ssim


class LPIPS(nn.Module):
    def __init__(self, boundary_ignore=None, type='alex', bgr2rgb=False):
        super().__init__()
        self.boundary_ignore = boundary_ignore
        self.bgr2rgb = bgr2rgb

        if type == 'alex':
            self.loss = lpips.LPIPS(net='alex')
        elif type == 'vgg':
            self.loss = lpips.LPIPS(net='vgg')
        else:
            raise Exception

    def forward(self, pred, gt, valid=None):
        if self.bgr2rgb:
            pred = pred[..., [2, 1, 0], :, :].contiguous()
            gt = gt[..., [2, 1, 0], :, :].contiguous()

        if self.boundary_ignore is not None:
            pred = pred[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        loss = self.loss(pred, gt)

        return loss.mean()


class AlignedL2(nn.Module):
    """ Computes L2 error after performing spatial and color alignment of the input image to GT"""
    def __init__(self, alignment_net, sr_factor=4, boundary_ignore=None):
        super().__init__()
        self.sca = SpatialColorAlignment(alignment_net, sr_factor)
        self.boundary_ignore = boundary_ignore

    def forward(self, pred, gt, burst_input):
        pred_warped_m, valid = self.sca(pred, gt, burst_input)

        # Ignore boundary pixels if specified
        if self.boundary_ignore is not None:
            pred_warped_m = pred_warped_m[..., self.boundary_ignore:-self.boundary_ignore,
                            self.boundary_ignore:-self.boundary_ignore]
            gt = gt[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

            valid = valid[..., self.boundary_ignore:-self.boundary_ignore, self.boundary_ignore:-self.boundary_ignore]

        # Estimate MSE
        mse = F.mse_loss(pred_warped_m, gt, reduction='none')

        eps = 1e-12
        elem_ratio = mse.numel() / valid.numel()
        mse = (mse * valid.float()).sum() / (valid.float().sum()*elem_ratio + eps)

        return mse
    

# def align_output_to_target(input_img, reference, burst_input=None, spatial=True, color=True, sr_factor=4, device=None):
#     """
#     Aligns an input image to a reference image using spatial and/or color alignment.
#     
#     Args:
#         input_img (torch.Tensor): Input image to be aligned, shape [B, C, H, W]
#         reference (torch.Tensor): Reference image to align to, shape [B, C, H, W]
#         burst_input (torch.Tensor, optional): Burst input for color matching, shape [B, T, C, H/sr_factor, W/sr_factor]
#         spatial (bool): Whether to perform spatial alignment
#         color (bool): Whether to perform color alignment
#         sr_factor (int): Super-resolution factor
#         device (torch.device, optional): Device to use. If None, uses the device of input_img
#         
#     Returns:
#         torch.Tensor: Aligned image
#         torch.Tensor: Valid mask indicating reliable aligned pixels
#     """
#     if device is None:
#         device = input_img.device
#     
#     # If neither spatial nor color alignment is requested, return the input
#     if not spatial and not color:
#         valid = torch.ones_like(input_img[:, 0:1])
#         return input_img, valid
#     
#     # Initialize PWCNet for alignment if spatial alignment is requested
#     from pwcnet import PWCNet
#     pwcnet = PWCNet(weights_path="pretrained_networks/pwcnet-network-default.pth").to(device)
# 
#     
#     # Initialize the SpatialColorAlignment module
#     sca = SpatialColorAlignment(pwcnet, sr_factor=sr_factor).to(device)
#     
#     # Perform alignment
#     with torch.no_grad():
#         aligned_img, valid = sca(input_img, reference, burst_input)
#     
#     return aligned_img, valid


def calculate_aligned_metrics(pred, target, burst_input=None, spatial=True, color=True, sr_factor=4, 
                             boundary_ignore=0, device=None):
    """
    Calculates metrics after aligning the prediction to the target.
    
    Args:
        pred (torch.Tensor): Prediction image, shape [B, C, H, W]
        target (torch.Tensor): Target image, shape [B, C, H, W]
        burst_input (torch.Tensor, optional): Burst input for color matching
        spatial (bool): Whether to perform spatial alignment
        color (bool): Whether to perform color alignment
        sr_factor (int): Super-resolution factor
        boundary_ignore (int): Number of boundary pixels to ignore in metrics
        device (torch.device, optional): Device to use
        
    Returns:
        dict: Dictionary containing metrics (PSNR, MSE)
    """
    valid = None
    if device is None:
        device = pred.device
    
    # Align prediction to target
    aligned_pred, valid = align_output_to_target(
        pred, target, burst_input, spatial, color, sr_factor, device
    )
    
    # Apply boundary ignore if specified
    if boundary_ignore > 0:
        aligned_pred = aligned_pred[..., boundary_ignore:-boundary_ignore, boundary_ignore:-boundary_ignore]
        target = target[..., boundary_ignore:-boundary_ignore, boundary_ignore:-boundary_ignore]
        valid = valid[..., boundary_ignore:-boundary_ignore, boundary_ignore:-boundary_ignore]
    
    # Calculate MSE
    mse = F.mse_loss(aligned_pred, target, reduction='none').mean(dim=1)
    if valid is not None:
        mse = (mse * valid.squeeze(1).float()).sum() / (valid.float().sum() + 1e-8)
    else:
        mse = mse.mean()
    
    # Calculate PSNR
    psnr = 20 * torch.log10(torch.tensor(1.0, device=mse.device)) - 10.0 * torch.log10(mse + 1e-8)
    
    # Return metrics
    return {
        'psnr': psnr.item(),
        'mse': mse.item(),
        'aligned_pred': aligned_pred,
        'valid': valid
    }


def evaluate_predictions(pred_path, target_path, burst_input_path=None, spatial=True, color=True, 
                         sr_factor=4, boundary_ignore=40, device=None):
    """
    Evaluates predictions against targets using alignment-based metrics.
    
    Args:
        pred_path (str): Path to prediction image
        target_path (str): Path to target image
        burst_input_path (str, optional): Path to burst input for color matching
        spatial (bool): Whether to perform spatial alignment
        color (bool): Whether to perform color alignment
        sr_factor (int): Super-resolution factor
        boundary_ignore (int): Number of boundary pixels to ignore in metrics
        device (torch.device, optional): Device to use
        
    Returns:
        dict: Dictionary containing metrics and visualizations
    """
    from PIL import Image
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import math
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading prediction from: {pred_path}")
    print(f"Loading target from: {target_path}")
    if burst_input_path:
        print(f"Loading burst input from: {burst_input_path}")
    
    # Check if files exist
    for path in [pred_path, target_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    if burst_input_path and not os.path.exists(burst_input_path):
        print(f"Warning: Burst input file not found: {burst_input_path}")
        burst_input_path = None
    
    # Load images
    pred_img = Image.open(pred_path).convert('RGB')
    target_img = Image.open(target_path).convert('RGB')


    
    # Convert to tensors
    to_tensor = lambda x: torch.from_numpy(np.array(x).transpose(2, 0, 1) / 255.0).float().unsqueeze(0).to(device)
    pred = to_tensor(pred_img)
    target = to_tensor(target_img)

    print(f"Prediction shape: {pred.shape}")
    print(f"Target shape: {target.shape}")

    # Initialize specialized metrics if available
    metrics = ('psnr', 'ssim', 'lpips')
    metrics_all = {}
    metrics_all['psnr'] = PSNR(boundary_ignore=boundary_ignore).to(device)
    metrics_all['ssim'] = SSIM(boundary_ignore=boundary_ignore, use_for_loss=False).to(device)
    metrics_all['lpips'] = LPIPS(boundary_ignore=boundary_ignore).to(device)
    
    # Calculate metrics before alignment
    print("Calculating metrics before alignment...")
    # Create a dummy valid mask (all ones)
    valid_before = torch.ones_like(pred[:, 0:1])
    metrics_before = {}
    for m, m_fn in metrics_all.items():
        metrics_before[m] = m_fn(pred, target, valid=valid_before).cpu().item()
    psnr_before = metrics_before['psnr']

    
    # Load burst input
    # Load the burst input as a single frame
    burst_img = Image.open(burst_input_path).convert('RGB')
    burst_single = to_tensor(burst_img)
    
    # Resize to match expected LR size
    if burst_single.shape[-1] != pred.shape[-1] // sr_factor:
        burst_size = pred.shape[-1] // sr_factor
        burst_single = F.avg_pool2d(burst_single, kernel_size=sr_factor, stride=sr_factor)

    # Initialize PWCNet for alignment
    #from pwcnet import PWCNet
    #print("Loading PWCNet for alignment...")
    #pwcnet = PWCNet(weights_path="pretrained_networks/pwcnet-network-default.pth").to(device)
    
    # Initialize the SpatialColorAlignment module
    #print("Initializing SpatialColorAlignment...")
    #sca = SpatialColorAlignment(pwcnet, sr_factor=sr_factor).to(device)
    
    # Perform alignment
    print("Performing alignment...")
    with torch.no_grad():
        # Use custom alignment if available
        print("Using custom alignment...")

        gauss_kernel, ksz = get_gaussian_kernel(sd=1.5)

        #aligned_pred = align_output_to_target(pred, target, spectral=True, spatial=False)
        pred = align_kornia_brute_force(pred.squeeze(0), target.squeeze(0)).unsqueeze(0)
        aligned_pred, _ = match_colors(pred, target, pred, ksz, gauss_kernel)
        valid = torch.ones_like(aligned_pred[:, 0:1])
        
        # Ensure valid mask exists
        if valid is None:
            print("Valid mask is None, creating a new one...")
            valid = torch.ones_like(aligned_pred[:, 0:1])
        elif valid.float().mean().item() < 0.1:
            print(f"Valid mask too small ({valid.float().mean().item()*100:.2f}%), using all pixels")
            valid = torch.ones_like(aligned_pred[:, 0:1])
        
    # Calculate metrics after alignment
    metrics_after = {}
    for m, m_fn in metrics_all.items():
        metrics_after[m] = m_fn(aligned_pred, target).cpu().item()
    psnr_aligned = metrics_after['psnr']
    
    print(f"PSNR before: {psnr_before:.4f}, after: {psnr_aligned:.4f}")

    
    # Create visualizations
    def tensor_to_np(tensor):
        return tensor.squeeze(0).permute(1, 2, 0).cpu().numpy().clip(0, 1)
    
    def visualize_flow(flow):
        """Visualize optical flow using color wheel."""
        # Convert flow to RGB image using HSV colorspace
        mag, ang = torch.hypot(flow[0,0], flow[0,1]), torch.atan2(flow[0,1], flow[0,0])
        
        # Normalize magnitude for better visualization
        mag = mag / (mag.max() + 1e-8)
        
        # Convert to HSV
        hsv = torch.zeros(3, flow.shape[2], flow.shape[3], device=flow.device)
        hsv[0] = (ang + torch.pi) / (2 * torch.pi)  # Hue from angle
        hsv[1] = torch.ones_like(ang)  # Full saturation
        hsv[2] = mag  # Value from magnitude
        
        # Convert HSV to RGB (simplified implementation)
        h, s, v = hsv[0], hsv[1], hsv[2]
        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c
        
        zeros = torch.zeros_like(h)
        
        # HSV to RGB conversion
        r = torch.zeros_like(h)
        g = torch.zeros_like(h)
        b = torch.zeros_like(h)
        
        # H in [0, 1/6)
        mask = h < 1/6
        r += torch.where(mask, c, zeros)
        g += torch.where(mask, x, zeros)
        
        # H in [1/6, 2/6)
        mask = (h >= 1/6) & (h < 2/6)
        r += torch.where(mask, x, zeros)
        g += torch.where(mask, c, zeros)
        
        # H in [2/6, 3/6)
        mask = (h >= 2/6) & (h < 3/6)
        g += torch.where(mask, c, zeros)
        b += torch.where(mask, x, zeros)
        
        # H in [3/6, 4/6)
        mask = (h >= 3/6) & (h < 4/6)
        g += torch.where(mask, x, zeros)
        b += torch.where(mask, c, zeros)
        
        # H in [4/6, 5/6)
        mask = (h >= 4/6) & (h < 5/6)
        r += torch.where(mask, x, zeros)
        b += torch.where(mask, c, zeros)
        
        # H in [5/6, 1)
        mask = h >= 5/6
        r += torch.where(mask, c, zeros)
        b += torch.where(mask, x, zeros)
        
        rgb = torch.stack([r + m, g + m, b + m], dim=0)
        return rgb.permute(1, 2, 0).cpu().numpy()
    
    # Create visualization figure
    print("Creating visualizations...")
    plt.figure(figsize=(15, 12))
    
    # First row: Original images
    plt.subplot(331)
    plt.imshow(tensor_to_np(pred))
    plt.title(f'Original Prediction\nPSNR: {psnr_before:.2f}')
    plt.axis('off')
    
    plt.subplot(332)
    plt.imshow(tensor_to_np(target))
    plt.title('Target')
    plt.axis('off')
    
    plt.subplot(333)
    diff_before = torch.abs(pred - target)
    plt.imshow(tensor_to_np(diff_before * 5))  # Amplify difference for visibility
    plt.title('Error Before Alignment (×5)')
    plt.axis('off')
    
    # Second row: Aligned images
    plt.subplot(334)
    plt.imshow(tensor_to_np(aligned_pred))
    plt.title(f'Aligned Prediction\nPSNR: {psnr_aligned:.2f}')
    plt.axis('off')
    
    plt.subplot(335)
    plt.imshow(tensor_to_np(target))
    plt.title('Target')
    plt.axis('off')
    
    plt.subplot(336)
    diff_after = torch.abs(aligned_pred - target)
    plt.imshow(tensor_to_np(diff_after * 5))  # Amplify difference for visibility
    plt.title('Error After Alignment (×5)')
    plt.axis('off')
    
    # Third row: Difference visualization
    plt.subplot(337)
    # Create a heatmap showing improvement
    improvement_map = diff_before - diff_after
    # Use a diverging colormap: blue for improvement, red for degradation
    plt.imshow(tensor_to_np(improvement_map * 5), cmap='RdBu')
    plt.title('Improvement Map (×5)\nBlue = Better, Red = Worse')
    plt.axis('off')
    
    plt.subplot(338)
    # Show the absolute improvement
    abs_improvement = torch.abs(improvement_map)
    plt.imshow(tensor_to_np(abs_improvement * 5), cmap='viridis')
    plt.title('Magnitude of Change (×5)')
    plt.axis('off')
    
    plt.subplot(339)
    # Create a binary mask showing where alignment improved things
    # Fix: Convert to proper format for display
    improved_mask = (diff_after < diff_before).float()
    # Take mean across channels to get a single-channel mask
    improved_mask_display = improved_mask.mean(dim=1, keepdim=True)[0, 0].cpu().numpy()
    plt.imshow(improved_mask_display, cmap='gray')
    plt.title(f'Improved Regions\n{improved_mask.mean().item()*100:.1f}% of pixels')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('alignment_comparison.png', dpi=150)
    print("Detailed alignment comparison saved as 'alignment_comparison.png'")
    
# Prepare return values
    metrics_dict = {
        'before': metrics_before,
        'after': metrics_after,
        'improvement': {m: metrics_after[m] - metrics_before[m] for m in metrics}
    }

    
    return {
        'metrics': metrics_dict,
        'visualizations': {
            'pred': tensor_to_np(pred),
            'target': tensor_to_np(target),
            'aligned_pred': tensor_to_np(aligned_pred),
            'error_before': tensor_to_np(diff_before),
        }
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate image predictions with alignment')
    parser.add_argument('--pred', type=str, required=False, help='Path to prediction image')
    parser.add_argument('--target', type=str, required=False, help='Path to target image')
    parser.add_argument('--burst', type=str, default=None, help='Path to burst input (optional)')
    parser.add_argument('--spatial', type=bool, default=True, help='Perform spatial alignment')
    parser.add_argument('--color', type=bool, default=True, help='Perform color alignment')
    parser.add_argument('--sr_factor', type=int, default=4, help='Super-resolution factor')
    parser.add_argument('--boundary_ignore', type=int, default=40, help='Boundary pixels to ignore')
    
    args = parser.parse_args()

    folder = "results/burst_synth/df4_shift1.0_samples16"
    import os
    
    # Initialize dictionaries to store accumulated metrics
    all_metrics_before = {}
    all_metrics_after = {}
    all_metrics_improvement = {}
    count = 0
    
    # Process each image in the folder
    for image_id in sorted(os.listdir(folder)):
        image_path = f"{folder}/{image_id}/mlp_fourier_3.0_2000_lr0.002_d4_h256"
        
        # Skip if required files don't exist
        if not os.path.exists(f"{image_path}/output_prediction_final.png") or \
           not os.path.exists(f"{image_path}/hr_ground_truth.png") or \
           not os.path.exists(f"{image_path}/lr_input.png"):
            print(f"Skipping {image_id} - missing required files")
            continue
            
        args.pred = f"{image_path}/output_prediction_final.png"
        args.target = f"{image_path}/hr_ground_truth.png"
        args.burst = f"{image_path}/lr_input.png"
        
        print(f"\n{'='*80}")
        print(f"Processing image {image_id} ({count+1})")
        print(f"{'='*80}")

        # Run evaluation
        results = evaluate_predictions(
            pred_path=args.pred, target_path=args.target, burst_input_path=args.burst,
            spatial=args.spatial, color=args.color,
            sr_factor=args.sr_factor, boundary_ignore=args.boundary_ignore
        )
        
        # Accumulate metrics
        for metric, value in results['metrics']['before'].items():
            if metric not in all_metrics_before:
                all_metrics_before[metric] = 0.0
            all_metrics_before[metric] += value
            
        for metric, value in results['metrics']['after'].items():
            if metric not in all_metrics_after:
                all_metrics_after[metric] = 0.0
            all_metrics_after[metric] += value
            
        for metric, value in results['metrics']['improvement'].items():
            if metric not in all_metrics_improvement:
                all_metrics_improvement[metric] = 0.0
            all_metrics_improvement[metric] += value
            
        count += 1
        
    
    # Calculate and display averages
    if count > 0:
        print(f"\n{'='*80}")
        print(f"AVERAGE METRICS ACROSS {count} IMAGES")
        print(f"{'='*80}")
        print(f"{'Metric':<10} {'Before Alignment':<20} {'After Alignment':<20} {'Improvement':<10}")
        print(f"{'-'*60}")
        
        for metric in all_metrics_before.keys():
            avg_before = all_metrics_before[metric] / count
            avg_after = all_metrics_after[metric] / count
            avg_improvement = all_metrics_improvement[metric] / count
            
            print(f"{metric.upper():<10} {avg_before:<20.4f} {avg_after:<20.4f} {avg_improvement:<10.4f}")
        
        # Save the average metrics to a file
        with open('average_metrics.txt', 'w') as f:
            f.write(f"AVERAGE METRICS ACROSS {count} IMAGES\n")
            f.write(f"{'Metric':<10} {'Before Alignment':<20} {'After Alignment':<20} {'Improvement':<10}\n")
            f.write(f"{'-'*60}\n")
            
            for metric in all_metrics_before.keys():
                avg_before = all_metrics_before[metric] / count
                avg_after = all_metrics_after[metric] / count
                avg_improvement = all_metrics_improvement[metric] / count
                
                f.write(f"{metric.upper():<10} {avg_before:<20.4f} {avg_after:<20.4f} {avg_improvement:<10.4f}\n")
        
        print(f"\nAverage metrics saved to 'average_metrics.txt'")
    else:
        print("No images were successfully processed.")