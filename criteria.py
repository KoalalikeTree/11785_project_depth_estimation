import torch
import torch.nn as nn
from math import exp
import torch.nn.functional as F
import numpy as np

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss

class SsimLoss(nn.Module):
    def __init__(self):
        super(SsimLoss, self).__init__()

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    def forward(self, img1, img2, val_range=1000.0/10.0, window_size=11, window=None, size_average=True, full=False):
        assert img2.dim() == img1.dim(), "inconsistent dimensions"
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).to(img1.device)

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

        # edit by cora

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        L_ssim = ( 1- ret )/2

        if full:
            return L_ssim, cs

        return L_ssim


class Ssim_grad_L1(nn.Module):
    def __init__(self):
        super(Ssim_grad_L1, self).__init__()

    def forward(self, y_true, y_pred, theta=0.1, maxDepthVal=1000.0/10.0):
        assert y_true.dim() == y_pred.dim(), "inconsistent dimensions"

        y_pred_temp = y_pred.cpu().detach().numpy()
        y_true_temp = y_true.cpu().detach().numpy()

        # Point-wise depth
        l_depth = np.mean(np.abs(y_pred_temp - y_true_temp), axis=-1)

        # Edges
        # dy_true, dx_true = tf.image.image_gradients(y_true)
        # dy_pred, dx_pred = tf.image.image_gradients(y_pred)

        dx_true, dy_true= np.gradient(y_true_temp, axis=(2,3))
        dx_pred, dy_pred= np.gradient(y_pred_temp, axis=(2,3))
        l_edges = np.mean(np.abs(dy_pred - dy_true) + np.abs(dx_pred - dx_true), axis=-1)
        # print("l_edges", l_edges)
        # print("l_depth", l_depth)

        # Structural similarity (SSIM) index

        l_ssim = torch.clamp((1 - self.ssim(y_pred, y_true, val_range = 1000.0 / 10.0)) * 0.5, 0, 1)
        # print("l_ssim=", l_ssim)

        # Weights
        w1 = 1.0
        w2 = 1.0
        w3 = theta

        return (w1 * l_ssim) + (w2 * np.mean(l_edges)) + (w3 * np.mean(l_depth))

    def ssim(self, img1, img2, val_range=1000.0/10.0, window_size=11, window=None, size_average=True, full=False):
        assert img2.shape == img1.shape, "inconsistent dimensions"
        L = val_range

        padd = 0
        (_, channel, height, width) = img1.shape
        if window is None:
            real_size = min(window_size, height, width)
            window = self.create_window(real_size, channel=channel).cuda()

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


        # edit by cora

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs

        return ret

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel=1):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss