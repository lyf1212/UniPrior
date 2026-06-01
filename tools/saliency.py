import cv2
import torch
import torch.nn.functional as F
import math
import torchvision

def gaussian_filter(kernel_size=5, sigma=1.0, channels=3):
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.T
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()
    
    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.
    
    kernel = (1. / (2. * math.pi * variance)) * \
             torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))
    
    kernel = kernel / torch.sum(kernel)
    
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)

    filter = torch.nn.Conv2d(3, 3, kernel_size=kernel_size, padding=kernel_size//2, bias=False, padding_mode='reflect', groups=3)
    filter.weight.data = kernel
    
    return filter


def calculate_saliency(x: torch.Tensor, ksz=5, sigma=1.0):
    '''
        return (b,c,h*w) for key point samping probability.
    '''
    b,c,h,w = x.shape
    x_freq = torch.fft.fft2(x)
    amplitude = torch.abs(x_freq)
    phase = torch.angle(x_freq)
    log_amplitude = torch.log(amplitude + 1e-6)
    blur_filter = gaussian_filter(ksz, sigma).to(x.device)
    avg_spectrum = blur_filter(log_amplitude)
    spectral_residual = log_amplitude - avg_spectrum
    reconstructed = torch.exp(spectral_residual + 1j*phase)
    saliency = torch.abs(torch.fft.ifft2(reconstructed)).mean(1)  # per channel filter, sum up to 1 channel finally.

    # below is post-processing.
    saliency = saliency.reshape(b, -1)
    # print(torch.quantile(saliency, q=0.95, dim=-1, keepdim=True).shape)
    saliency[saliency < torch.quantile(saliency, q=0.95, dim=-1, keepdim=True)] = 0
    saliency = F.softmax(saliency, dim=-1)
    saliency = F.interpolate(saliency.reshape(b,1,h,w), size=(37, 37))
    
    saliency_vis = (saliency - saliency.min()) / (saliency.max() - saliency.min())
    torchvision.utils.save_image(saliency_vis, './debug_saliency.png')

    return saliency.reshape(b,-1)


if __name__ == '__main__':
    x = cv2.imread('/mnt/netdisk/liyf/DAI-Net/DG/classification/n02835271_10064.JPEG').transpose(2,0,1) / 255.
    x = torch.tensor(x).unsqueeze(0).type(torch.float32)
    x = torch.concat([x, x], dim=0)
    print(x.shape)
    saliency = calculate_saliency(x)
    print(saliency.shape)
    print(torch.multinomial(saliency, 1))