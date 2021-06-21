import numpy as np
from skimage.metrics import structural_similarity as comp_ssim
import lpips
import glob
import os
import imageio
import torch
import math

def calc_ssim( hr, sr, align=False):
    sr = np.transpose(sr[0].cpu().numpy(), (1,2,0))
    hr = np.transpose(hr[0].cpu().numpy(), (1,2,0))
    return comp_ssim(sr/255., hr/255., multichannel=True)

def calc_psnr(hr, sr, scale=1, rgb_range=255, align=False, dataset=None):
    if hr.nelement() == 1: return 0
    diff = (sr - hr) / rgb_range
    mse = diff.pow(2).mean()
    return -10 * math.log10(mse)

def calc_lpips(hr, sr, loss_fn):
    sr = (sr / 255.) * 2- 1 
    hr = (hr / 255.) * 2- 1
    d = loss_fn(sr, hr)[0,0,0,0]
    return d

def np2Tensor(*args):
    def _np2Tensor(img):
        np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))
        np_transpose = np.expand_dims(np_transpose, 0)
        tensor = torch.from_numpy(np_transpose).float()
        return tensor

    return [_np2Tensor(a) for a in args]

def Mean(lst): 
    return sum(lst) / len(lst)

def compute_all(root):
    all_file = glob.glob(os.path.join(root, 'target', '*GT.png'))
    psnr = []
    ssim = []
    lpips_score = []
    loss_fn_alex = lpips.LPIPS(net='alex')
    for file_name in all_file:
        print(file_name)
        hr = imageio.imread(file_name)
        sr = imageio.imread(file_name.replace('/target/','/transfill/').replace('GT','Final'))
        hr, sr = np2Tensor(hr, sr)
        psnr.append(calc_psnr(hr, sr))
        ssim.append(calc_ssim(hr, sr))
        lpips_score.append(calc_lpips(hr, sr, loss_fn_alex))
    return Mean(psnr), Mean(ssim), Mean(lpips_score)

root = 'data/Small_Set'
print(compute_all(root))
