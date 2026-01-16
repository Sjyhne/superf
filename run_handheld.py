# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 13:51:42 2023

@author: jamyl
"""

import os
import glob

import argparse
from pathlib import Path
import numpy as np
import torch
from skimage import img_as_float32, img_as_ubyte, io, color, filters
import cv2
import rawpy

from handheld.super_resolution import process
from handheld.utils_dng import save_as_dng


def color_match_mean_std(pred, gt, eps=1e-8, clip=(0.0, 1.0)):
    """
    Per-channel affine: y ≈ s*x + b, with s=std_gt/std_pred, b=mean_gt - s*mean_pred.
    """
    p = pred.reshape(-1, 3).astype(np.float32)
    g = gt.reshape(-1, 3).astype(np.float32)

    mp, sp = p.mean(axis=0), p.std(axis=0) + eps
    mg, sg = g.mean(axis=0), g.std(axis=0)

    s = sg / sp
    b = mg - s * mp

    out = pred * s + b  # broadcast
    if clip is not None:
        out = np.clip(out, *clip)
    return out, np.diag(s), b


def linear_color_match(
    pred_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    mask: np.ndarray | None = None,
    add_bias: bool = True,
    ridge: float = 1e-3,
    robust_trim_percentile: float | None = 95.0,
    clip_range: tuple[float, float] | None = (0.0, 1.0),
):
    """
    Fit a global linear color transform that maps pred_rgb -> gt_rgb.

    Args:
        pred_rgb:  HxWx3 float array (prediction in linear RGB).
        gt_rgb:    HxWx3 float array (ground truth in linear RGB), spatially aligned to pred_rgb.
        mask:      Optional HxW boolean array of valid pixels (e.g., from flow/confidence).
        add_bias:  If True, fit Y ≈ A*X + b (3x3 + bias); else fit Y ≈ A*X (3x3).
        ridge:     L2 regularization strength (λ) for stability.
        robust_trim_percentile:
                   If set (e.g., 95), do a two-pass fit:
                     1) fit on all valid pixels,
                     2) compute residuals, keep pixels below given percentile, refit.
                   Set to None to disable trimming.
        clip_range: If not None, clip the corrected image to this range.

    Returns:
        corrected_rgb: HxWx3 array = color-corrected pred_rgb.
        A:             3x3 matrix.
        b:             3-vector (zeros if add_bias=False).
        inlier_mask:   HxW boolean mask actually used in the final fit.
    """
    assert pred_rgb.shape == gt_rgb.shape and pred_rgb.shape[-1] == 3
    H, W, _ = pred_rgb.shape

    # Build initial valid mask
    valid = np.isfinite(pred_rgb).all(axis=-1) & np.isfinite(gt_rgb).all(axis=-1)
    if mask is not None:
        valid &= mask.astype(bool)

    # Flatten valid pixels
    X = pred_rgb[valid].reshape(-1, 3)
    Y = gt_rgb[valid].reshape(-1, 3)
    
    # Ensure X and Y have the same number of pixels
    assert X.shape[0] == Y.shape[0], f"X and Y must have same number of pixels: X={X.shape[0]}, Y={Y.shape[0]}"

    def _fit(X_, Y_):
        # Ensure X_ and Y_ have the same number of samples
        assert X_.shape[0] == Y_.shape[0], f"X_ and Y_ must have same number of samples: X_={X_.shape[0]}, Y_={Y_.shape[0]}"
        
        # Design matrix: [X | 1] if bias, else [X]
        if add_bias:
            ones = np.ones((X_.shape[0], 1), dtype=X_.dtype)
            DM = np.hstack([X_, ones])       # N x 4
            I = np.eye(4, dtype=X_.dtype)
        else:
            DM = X_                           # N x 3
            I = np.eye(3, dtype=X_.dtype)

        # Ridge LS solve for each channel jointly: W = (DM^T DM + λI)^-1 DM^T Y
        # W has shape (4x3) if bias else (3x3); last row is bias if add_bias.
        XtX = DM.T @ DM
        W = np.linalg.solve(XtX + ridge * I, DM.T @ Y_)

        if add_bias:
            A = W[:3, :].T   # 3x3
            b = W[3, :]      # 3
        else:
            A = W.T          # 3x3
            b = np.zeros(3, dtype=X_.dtype)
        return A, b

    # First fit
    A, b = _fit(X, Y)

    # Optional robust trimming (discard high-residual pixels and refit)
    inlier_mask = valid.copy()
    if robust_trim_percentile is not None:
        Y_hat = (X @ A.T) + b  # N x 3
        resid = np.mean((Y_hat - Y) ** 2, axis=1)  # per-pixel MSE
        thresh = np.percentile(resid, robust_trim_percentile)
        keep = resid <= thresh
        
        # Refit on inliers using the same valid pixels
        X2 = X[keep]  # Use the same valid pixels that were kept
        Y2 = Y[keep]  # Use the same valid pixels that were kept
        if X2.shape[0] >= 16:  # minimal safety check
            A, b = _fit(X2, Y2)
            # Update inlier mask in image space
            inlier_idx = np.where(valid.ravel())[0]
            inlier_mask = valid.ravel().copy()
            inlier_mask[inlier_idx[~keep]] = False
            inlier_mask = inlier_mask.reshape(H, W)
        else:
            # Fallback: keep first fit if too few points
            inlier_mask = valid

    # Apply transform
    corrected = (pred_rgb @ A.T) + b

    if clip_range is not None:
        lo, hi = clip_range
        corrected = np.clip(corrected, lo, hi)

    return corrected, A, b, inlier_mask 


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    #### Argparser
    
    def str2bool(v):
        v = str(v)
    
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    parser = argparse.ArgumentParser()
    ## Image parameter
    parser.add_argument('--impath', type=str, help='input image')
    parser.add_argument('--outpath', type=str, help='out image')
    parser.add_argument('--scale', type=int, default=2, help='Scaling factor')
    parser.add_argument('--verbose', type=int, default=1, help='Verbose option (0 to 4)')
    
    ## Robustness
    parser.add_argument('--t', type=float,  default=0.12, help='Threshold for robustness')
    parser.add_argument('--s1', type=float,  default=2, help='Threshold for robustness')
    parser.add_argument('--s2', type=float,  default=12, help='Threshold for robustness')
    parser.add_argument('--Mt', type=float,  default=0.8, help='Threshold for robustness')
    parser.add_argument('--R_on', type=str2bool,  default=True, help='Whether robustness is activated or not')
    
    parser.add_argument('--R_denoising_on', type=str2bool, default=True, help='Whether or not the robustness based denoising should be applied')
    
    
    ## Post Processing
    parser.add_argument('--post_process', type=str2bool,  default=True, help='Whether post processing should be applied or not')
    parser.add_argument('--do_sharpening', type=str2bool,  default=True, help='Whether sharpening should be applied during post processing')
    parser.add_argument('--radius', type=float,  default=3, help='If sharpening is applied, radius of the unsharp mask')
    parser.add_argument('--amount', type=float,  default=1.5, help='If sharpening is applied, amount of the unsharp mask')
    parser.add_argument('--do_tonemapping', type=str2bool,  default=True, help='Whether tonnemaping should be applied during post processing')
    parser.add_argument('--do_gamma', type=str2bool,  default=True, help='Whether gamma curve should be applied during post processing')
    parser.add_argument('--do_color_correction', type=str2bool,  default=True, help='Whether color correction should be applied during post processing')
    
    
    ## Merging (advanced)
    parser.add_argument('--kernel_shape', type=str, default='handheld', help='"handheld" or "iso" : Whether to use steerable or isotropic kernels')
    parser.add_argument('--k_detail', type=float, default=None, help='SNR based by default')
    parser.add_argument('--k_denoise', type=float, default=None, help='SNR based by default')
    parser.add_argument('--k_stretch', type=float, default=4)
    parser.add_argument('--k_shrink', type=float, default=2)
    
    ## Alignment (advanced)
    parser.add_argument('--ICA_iter', type=int, default=3, help='Number of ICA Iterations')
    
    ## Color matching
    parser.add_argument('--color_match', type=str2bool, default=False, help='Whether to apply color matching to the output')
    parser.add_argument('--reference_image', type=str, default=None, help='Reference image for color matching (optional)')
    
    
    args = parser.parse_args()
    
    
    
    print('Parameters:')
    print('')
    print('  Upscaling factor:       %d' % args.scale)
    print('')
    if args.scale == 1:
        print('    Demosaicking mode')
    else:
        print('    Super-resolution mode.')
        if args.scale > 2:
            print('    WARNING: Since the optics and the integration on the sensor limit the aliasing, do not expect more details than that obtained at x2 (refer to our paper and the original publication).')
    print('')
    if args.R_on:
        print('  Robustness:       enabled')
        print('  -------------------------')
        print('  t:                      %1.2f' % args.t)
        print('  s1:                     %1.2f' % args.s1)
        print('  s2:                     %1.2f' % args.s2)
        print('  Mt:                     %1.2f' % args.Mt)
        if args.R_denoising_on:
            print('  Robustness denoising:   enabled')
        else:
            print('  Robustness denoising:   disabled')
        print('                            ' )
    else:
        print('  Robustness:      disabled')
        print('                            ' )
    
    print('  Alignment:')
    print('  -------------------------')
    print('  ICA Iterations:         %d' % args.ICA_iter)
    print('')
    print('  Fusion:')
    print('  -------------------------')
    print('  Kernel shape:           %s' % args.kernel_shape)
    print('  k_stretch:              %1.2f' % args.k_stretch)
    print('  k_shrink:               %1.2f' % args.k_shrink)
    if args.k_detail is not None:
        print('  k_detail:               %1.2f' % args.k_detail)
    else:
        print('  k_detail:               SNR based' )
    if args.k_denoise is not None:
        print('  k_denoise:              %1.2f' % args.k_denoise)
    else:
        print('  k_denoise:              SNR based' )
    print('')
    print('  Color Matching:')
    print('  -------------------------')
    if args.color_match:
        print('  Color matching:         enabled')
        if args.reference_image:
            print(f'  Reference image:        {args.reference_image}')
        else:
            print('  Reference image:        not provided (will skip)')
    else:
        print('  Color matching:         disabled')
    print('')
    
    
    
    
    
    #### Handheld ####
    print('Processing with handheld super-resolution')
    options = {'verbose' : args.verbose}
    params={"scale": args.scale,
            "merging":{"kernel":args.kernel_shape},
            "robustness":{"on":args.R_on},
            "kanade": {"tuning": {"kanadeIter":args.ICA_iter}}
            }
    
    
    params['robustness']['tuning'] = {'t' : args.t,
                                      's1' : args.s1,
                                      's2' : args.s2,        
                                      'Mt' : args.Mt,       
                                      }
   
    params['merging'] = {'tuning' : {'k_stretch' : args.k_stretch,
                                     'k_shrink' : args.k_shrink
                                     }}

    if args.k_detail is not None:
        params['merging']['tuning']['k_detail'] = args.k_detail
    if args.k_denoise is not None:
        params['merging']['tuning']['k_denoise'] = args.k_denoise
    
    params['accumulated robustness denoiser'] = {'on': args.R_denoising_on}
    
    outpath = Path(args.outpath)
    # disabling post processing for dng outputs
    if outpath.suffix == '.dng':
        args.post_process = False
    
    
    params['post processing'] = {'on':args.post_process,
                        'do sharpening' : args.do_sharpening,
                        'do tonemapping':args.do_tonemapping,
                        'do gamma' : args.do_gamma,
                        'do devignette' : False,
                        'do color correction': args.do_color_correction,
            
                        'sharpening' : {'radius': args.radius,
                                        'ammount': args.amount}
                        }
    
    
    handheld_output = process(args.impath, options, params)
    handheld_output = np.nan_to_num(handheld_output)
    handheld_output = np.clip(handheld_output, 0, 1)
    
    # Apply color matching if requested
    if args.color_match:
        if args.reference_image is not None:
            # Load reference image for color matching
            try:
                ref_img = io.imread(args.reference_image)
                if ref_img.ndim == 3 and ref_img.shape[2] == 3:
                    ref_img = img_as_float32(ref_img)
                    # Resize reference to match handheld output if needed
                    if ref_img.shape[:2] != handheld_output.shape[:2]:
                        ref_img = cv2.resize(ref_img, (handheld_output.shape[1], handheld_output.shape[0]), 
                                           interpolation=cv2.INTER_LINEAR)
                    
                    print("Applying color matching with reference image...")
                    handheld_output, _, _, _ = linear_color_match(
                        handheld_output, ref_img, 
                        robust_trim_percentile=95.0, clip_range=(0.0, 1.0)
                    )
                    handheld_output = np.clip(handheld_output, 0, 1)
                    print("Color matching completed.")
                else:
                    print("Warning: Reference image must be RGB. Skipping color matching.")
            except Exception as e:
                print(f"Warning: Could not load reference image for color matching: {e}")
                print("Skipping color matching.")
        else:
            print("Warning: Color matching requested but no reference image provided.")
            print("Skipping color matching.")
    
    
    # define a faster imsave for large png images
    def imsave(fname, rgb_8bit_data):
        return cv2.imwrite(fname,  cv2.cvtColor(rgb_8bit_data, cv2.COLOR_RGB2BGR ))
    
    
    #### Save images ####
    
    if outpath.suffix == '.dng':
        if options['verbose'] >=1 :
            print('Saving output to {}'.format(outpath.with_suffix('.dng').as_posix()))
        ref_img_path = glob.glob(os.path.join(args.impath, '*.dng'))[0]
        save_as_dng(handheld_output, ref_img_path, outpath)
        
    else:
        imsave(args.outpath, img_as_ubyte(handheld_output))